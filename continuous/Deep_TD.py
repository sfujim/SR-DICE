import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		return self.l3(q1)


class Deep_TD(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005
	):

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau

		self.total_it = 0

		self.max_action = max_action


	def train_OPE(self, replay_buffer, policy, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			next_action = policy(next_state)
			next_action = (next_action + torch.randn_like(next_action) * self.max_action * 0.1).clamp(-self.max_action, self.max_action)
			target_Q = reward + self.discount * not_done * self.critic_target(next_state, next_action)

		current_Q = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def eval_policy(self, replay_buffer, policy, batch_size=10000):
		start_state = replay_buffer.all_start()
		start_action = policy(start_state)
		start_action = (start_action + torch.randn_like(start_action) * self.max_action * 0.1).clamp(-self.max_action, self.max_action)

		R =  (1. - self.discount) * self.critic(start_state, start_action).mean()
		return float(R)