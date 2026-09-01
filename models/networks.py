
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# شبکه‌های مربوط به عامل‌های بخشی (Sector Agents - PPO)
# ============================================================

class SectorActor(nn.Module):
    """شبکه سیاست برای هر بخش (خروجی وزن‌های داخلی)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)   # جمع وزن‌ها = ۱
        )
    
    def forward(self, state):
        return self.net(state)


class SectorCritic(nn.Module):
    """شبکه ارزش برای هر بخش"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


# ============================================================
# شبکه‌های مربوط به MAC (SAC)
# ============================================================

class MACActor(nn.Module):
    """
    شبکه سیاست برای MAC
    خروجی: [وزن_نقدینگی, آلفای_بخش_۱, ..., آلفای_بخش_S]
    جمع کل = ۱
    """
    def __init__(self, state_dim, num_sectors, hidden_dim=256):
        super().__init__()
        self.num_sectors = num_sectors
        self.action_dim = num_sectors + 1  # +1 برای نقدینگی
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class MACCritic(nn.Module):
    """شبکه ارزش برای MAC (دو شبکه برای SAC)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DoubleMACCritic(nn.Module):
    """دو شبکه‌ی ارزش برای SAC (Clipped Double-Q)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.critic1 = MACCritic(state_dim, action_dim, hidden_dim)
        self.critic2 = MACCritic(state_dim, action_dim, hidden_dim)
    
    def forward(self, state, action):
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2
