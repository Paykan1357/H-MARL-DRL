
"""
sector_ppo_agent.py
عامل PPO برای یک بخش خاص (با مدیریت خطا و بارگذاری چک‌پوینت)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import SectorActor, SectorCritic

class SectorPPOAgent:
    """
    عامل PPO برای یک بخش خاص (مثلاً فناوری، مالی، ...)
    هر بخش یک نمونه از این کلاس دارد.
    """
    def __init__(self, sector_name, state_dim, action_dim, args=None):
        self.sector_name = sector_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters (قابل تنظیم)
        self.gamma = getattr(args, "ppo_gamma", 0.99)
        self.gae_lambda = getattr(args, "ppo_gae_lambda", 0.95)
        self.epsilon_clip = getattr(args, "ppo_epsilon_clip", 0.2)
        self.entropy_coef = getattr(args, "ppo_entropy_coef", 0.01)
        self.value_loss_coef = getattr(args, "ppo_value_loss_coef", 0.5)
        self.max_grad_norm = getattr(args, "ppo_max_grad_norm", 0.5)
        self.ppo_epochs = getattr(args, "ppo_epochs", 10)
        self.batch_size = getattr(args, "ppo_batch_size", 64)
        
        self.lr_actor = getattr(args, "ppo_lr_actor", 3e-4)
        self.lr_critic = getattr(args, "ppo_lr_critic", 1e-3)
        
        # شبکه‌ها
        self.actor = SectorActor(state_dim, action_dim).to(self.device)
        self.critic = SectorCritic(state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # بافر
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.checkpoint_dir = getattr(args, "checkpoint_dir", f"./checkpoints_ppo/{sector_name}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"[PPO-{sector_name}] Initialized (state_dim={state_dim}, action_dim={action_dim})")
    
    def get_action(self, state, deterministic=False):
        """
        دریافت عمل (وزن‌های داخلی بخش) با مدیریت کامل خطا برای State
        """
        # --- تبدیل امن State به Tensor (مقاوم در برابر رشته و داده‌های غیرعددی) ---
        if not isinstance(state, torch.Tensor):
            state = np.asarray(state).flatten()
            if state.dtype == np.object_ or state.dtype == np.str_:
                state = np.array([float(x) if isinstance(x, (int, float)) else 0.0 for x in state], dtype=np.float32)
            else:
                state = state.astype(np.float32)
            
            if state.shape[0] != self.state_dim:
                # اگر ابعاد تطابق نداشت، کوتاه یا padding کن
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                else:
                    pad = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
                    state = np.concatenate([state, pad])
            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.actor(state)
            if deterministic:
                action = probs.argmax(dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            value = self.critic(state)
        
        # ذخیره‌سازی برای به‌روزرسانی
        self.states.append(state.cpu().numpy().flatten())
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        # خروجی به‌صورت آرایه‌ای از وزن‌ها (برای محیط)
        weights = probs.squeeze().cpu().numpy()
        return weights, action.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        if len(self.states) < self.batch_size:
            return None
        
        # تبدیل به تنسور
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        
        # GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        with torch.no_grad():
            last_state = torch.tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_state).item()
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value * (1 - dones[t])
            else:
                next_value = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # PPO Epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor Loss
                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic Loss
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Total
                total_loss = actor_loss + self.value_loss_coef * critic_loss
                
                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        return {
            "actor_loss": total_actor_loss / n_updates if n_updates > 0 else 0,
            "critic_loss": total_critic_loss / n_updates if n_updates > 0 else 0,
            "entropy": total_entropy / n_updates if n_updates > 0 else 0,
        }
    
    def save(self, episode):
        """ذخیره چک‌پوینت"""
        torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/actor_ep{episode}.pth")
        torch.save(self.critic.state_dict(), f"{self.checkpoint_dir}/critic_ep{episode}.pth")
    
    def load(self, episode):
        """
        بارگذاری چک‌پوینت.
        در صورت عدم تطابق ابعاد، خطا نمی‌دهد و False برمی‌گرداند.
        """
        actor_path = f"{self.checkpoint_dir}/actor_ep{episode}.pth"
        critic_path = f"{self.checkpoint_dir}/critic_ep{episode}.pth"
        
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            return False
        
        try:
            # بارگذاری Actor
            actor_state = torch.load(actor_path, map_location=self.device)
            # بررسی تطابق ابعاد
            if any(self.actor.state_dict()[k].shape != v.shape for k, v in actor_state.items()):
                print(f"   ⚠️ Actor dimension mismatch for {self.sector_name}. Skipping load.")
                return False
            self.actor.load_state_dict(actor_state)
            
            # بارگذاری Critic
            critic_state = torch.load(critic_path, map_location=self.device)
            if any(self.critic.state_dict()[k].shape != v.shape for k, v in critic_state.items()):
                print(f"   ⚠️ Critic dimension mismatch for {self.sector_name}. Skipping load.")
                return False
            self.critic.load_state_dict(critic_state)
            
            return True
        except RuntimeError as e:
            print(f"   ⚠️ Could not load checkpoint for {self.sector_name}: {e}. Starting fresh.")
            return False
    
    def reset_buffer(self):
        """پاک کردن بافر تجربه"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
