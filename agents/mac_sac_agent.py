
"""
mac_sac_agent.py
عامل SAC برای MAC (شامل نقدینگی) با مدیریت ابعاد پویا
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import MACActor, DoubleMACCritic

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)


class MACSACAgent:
    """
    عامل SAC برای MAC (شامل نقدینگی) با پشتیبانی از ابعاد متغیر
    """
    def __init__(self, state_dim, num_sectors, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.num_sectors = num_sectors
        self.action_dim = num_sectors + 1  # +1 برای نقدینگی
        
        # Hyperparameters
        self.gamma = getattr(args, "sac_gamma", 0.99)
        self.tau = getattr(args, "sac_tau", 0.005)
        self.alpha = getattr(args, "sac_alpha", 0.2)
        self.batch_size = getattr(args, "sac_batch_size", 256)
        self.lr = getattr(args, "sac_lr", 3e-4)
        
        # شبکه‌ها
        self.actor = MACActor(state_dim, num_sectors).to(self.device)
        self.critic = DoubleMACCritic(state_dim, self.action_dim).to(self.device)
        self.critic_target = DoubleMACCritic(state_dim, self.action_dim).to(self.device)
        
        # کپی اولیه
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # بهینه‌سازها
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # بافر
        self.buffer = ReplayBuffer(capacity=100000)
        
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints_sac/mac")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"[SAC-MAC] Initialized (state_dim={state_dim}, action_dim={self.action_dim})")
    
    def get_action(self, state, deterministic=False):
        """دریافت عمل: [وزن_نقدینگی, آلفا_بخش_۱, ..., آلفا_بخش_S]"""
        # ✅ اطمینان از اینکه state یک آرایه‌ی عددی با ابعاد صحیح است
        if not isinstance(state, torch.Tensor):
            state = np.asarray(state, dtype=np.float32).flatten()
            # اگر ابعاد با state_dim تطابق نداشت، اصلاح کن
            if state.shape[0] != self.state_dim:
                print(f"⚠️ MAC state dimension mismatch: got {state.shape[0]}, expected {self.state_dim}. Resizing.")
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                else:
                    pad = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
                    state = np.concatenate([state, pad])
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state)
            action = action.squeeze().cpu().numpy()
        
        # اطمینان از جمع ۱
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action /= action.sum()
        else:
            action[0] = 1.0
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # ---- به‌روزرسانی Critic ----
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * (1 - dones) * target_q
        
        q1, q2 = self.critic(states, actions)
        loss_critic = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        # ---- به‌روزرسانی Actor ----
        new_actions = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        loss_actor = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        # ---- Soft Update ----
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "critic_loss": loss_critic.item(),
            "actor_loss": loss_actor.item(),
        }
    
    def save(self, episode):
        """ذخیره چک‌پوینت"""
        torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/actor_ep{episode}.pth")
        torch.save(self.critic.state_dict(), f"{self.checkpoint_dir}/critic_ep{episode}.pth")
        torch.save(self.critic_target.state_dict(), f"{self.checkpoint_dir}/critic_target_ep{episode}.pth")
    
    def load(self, episode):
        """
        بارگذاری چک‌پوینت با بررسی تطابق ابعاد.
        در صورت mismatch، False برمی‌گرداند و خطا نمی‌دهد.
        """
        actor_path = f"{self.checkpoint_dir}/actor_ep{episode}.pth"
        critic_path = f"{self.checkpoint_dir}/critic_ep{episode}.pth"
        critic_target_path = f"{self.checkpoint_dir}/critic_target_ep{episode}.pth"
        
        if not os.path.exists(actor_path):
            return False
        
        try:
            # بارگذاری Actor با بررسی تطابق ابعاد
            actor_state = torch.load(actor_path, map_location=self.device)
            # بررسی لایه اول (ورودی) برای تطابق state_dim
            first_key = list(actor_state.keys())[0]
            if actor_state[first_key].shape[1] != self.state_dim:
                print(f"   ⚠️ MAC state_dim mismatch: checkpoint has {actor_state[first_key].shape[1]}, current {self.state_dim}")
                return False
            
            # بررسی لایه آخر (خروجی) برای تطابق action_dim
            last_key = list(actor_state.keys())[-1]
            if actor_state[last_key].shape[0] != self.action_dim:
                print(f"   ⚠️ MAC action_dim mismatch: checkpoint has {actor_state[last_key].shape[0]}, current {self.action_dim}")
                return False
            
            self.actor.load_state_dict(actor_state)
            
            # بارگذاری Critic
            critic_state = torch.load(critic_path, map_location=self.device)
            first_key = list(critic_state.keys())[0]
            if critic_state[first_key].shape[1] != self.state_dim + self.action_dim:
                print(f"   ⚠️ MAC critic dimension mismatch")
                return False
            
            self.critic.load_state_dict(critic_state)
            self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=self.device))
            return True
            
        except Exception as e:
            print(f"   ⚠️ MAC checkpoint load error: {e}")
            return False
