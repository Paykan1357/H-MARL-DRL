
import random
import numpy as np
import torch
from torch_utils import FLOAT

class ReplayBuffer:
    """
    بافر تجربه برای SAC (و PPO در صورت نیاز)
    هر نمونه شامل (state, action, reward, next_state, done) است.
    state و next_state به‌صورت تاپل (prices, weights) ذخیره می‌شوند.
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.state_shape = state_shape  # (num_assets, window_length, features)

    def push(self, state, action, reward, next_state, done):
        # state و next_state به‌صورت تاپل (prices, weights) هستند
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        # استخراج prices و weights از state و next_state
        state_prices = [s[0] for s in states]
        state_weights = [s[1] for s in states]
        next_state_prices = [ns[0] for ns in next_states]
        next_state_weights = [ns[1] for ns in next_states]

        # تبدیل به تنسور و padding (اگر ابعاد متفاوت باشند)
        state_prices = self._pad_and_stack(state_prices)
        state_weights = torch.tensor(np.array(state_weights), dtype=torch.float32)
        next_state_prices = self._pad_and_stack(next_state_prices)
        next_state_weights = torch.tensor(np.array(next_state_weights), dtype=torch.float32)

        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)

        return (state_prices, state_weights), actions, rewards, dones, (next_state_prices, next_state_weights)

    def _pad_and_stack(self, prices_list):
        """هم‌سازی ابعاد prices و تبدیل به تنسور"""
        max_A = max(p.shape[0] for p in prices_list)
        max_T = max(p.shape[1] for p in prices_list)
        max_F = max(p.shape[2] for p in prices_list)

        padded = []
        for p in prices_list:
            pad_A = max_A - p.shape[0]
            pad_T = max_T - p.shape[1]
            pad_F = max_F - p.shape[2]
            p_pad = np.pad(p, ((0, pad_A), (0, pad_T), (0, pad_F)), mode='constant')
            padded.append(p_pad)
        return torch.tensor(np.array(padded), dtype=torch.float32)

    def __len__(self):
        return len(self.buffer)
