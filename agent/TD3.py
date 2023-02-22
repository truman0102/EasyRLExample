import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sample import ReplayBuffer
from utils.update import soft_update
from network.body import policy_net, TD3_Critic_Net


class TD3:
    def __init__(self, input_channels: int, width: int, action_dim: int, noisy: bool, gamma: float,
                 learning_rate: float, tau: float, batch_size: int, capacity: int, replace_interval: int,
                 checkpoint_dir='', policy_noise=0.2, noise_clip=0.5):
        super(TD3, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = policy_net(input_channels=input_channels, width=width, action_dim=action_dim, noisy=noisy).to(
            self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = TD3_Critic_Net(input_channels=input_channels, width=width, action_dim=action_dim, noisy=noisy).to(
            self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.memory = ReplayBuffer(capacity)
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.replace_interval = replace_interval
        self.learn_step_counter = 0

    def train(self, training: bool = True):
        if self.memory.__len__ < self.batch_size:
            return

        self.actor.training = training
        self.actor_target.training = training
        self.critic.training = training
        self.critic_target.training = training

        stage, action, reward, next_stage, done = self.memory.sample(self.batch_size)
        stage = torch.from_numpy(stage).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_stage = torch.from_numpy(next_stage).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            noise = (torch.rand_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_stage) + noise).clamp(-self.noise_clip, self.noise_clip)

            target_q1, target_q2 = self.critic_target(next_stage, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * target_q * (1 - done)  # done=1表示结束

        current_q1, current_q2 = self.critic(stage, action)
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learn_step_counter % self.replace_interval == 0:
            actor_loss = -self.critic.Q1(stage, self.actor(stage)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(target=self.actor_target, source=self.actor, tau=self.tau)
            soft_update(target=self.critic_target, source=self.critic, tau=self.tau)
