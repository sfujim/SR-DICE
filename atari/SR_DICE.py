import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_Decoder(nn.Module):
	def __init__(self, frames, num_actions, device):
		super(Encoder_Decoder, self).__init__()


		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, 1 * 256) #


		self.l3 = nn.Linear(256, 3136)
		self.d1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
		self.d2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
		self.d3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4)

		self.r1 = nn.Linear(256, 1, bias=False)

		self.num_actions = num_actions
		self.device = device


	def forward(self, state, action):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))

		q = q.reshape(-1, 3136)
		q = F.relu(self.l1(q))
		lat = F.relu(self.l2(q)).reshape(-1, 256)

		r = self.r1(lat)

		q = F.relu(self.l3(lat))
		q = F.relu(self.d1(q.reshape(-1, 64, 7, 7)))
		q = F.relu(self.d2(q))
		ns = self.d3(q)

		return ns.reshape(-1, 84, 84), r, lat


	def latent(self, state, action):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = q.reshape(-1, 3136)
		q = F.relu(self.l1(q))
		q = F.relu(self.l2(q)).reshape(-1, 256)

		return q


# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, num_actions * 256)

		self.num_actions = num_actions


	def forward(self, state):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = F.relu(self.l1(q.reshape(-1, 3136)))
		return self.l2(q).reshape(-1, self.num_actions, 256)


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)


class SR_DICE(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(4, num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.encoder_decoder = Encoder_Decoder(4, num_actions, device).to(self.device)
		self.ed_optimizer = getattr(torch.optim, optimizer)(self.encoder_decoder.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1, 4, 84, 84) if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0

		self.w = torch.ones(1, 256, requires_grad=True, device=device)
		self.w_optimizer = torch.optim.Adam([self.w], lr=3e-4)

		self.eps = 0
		

	def e_greedy(self, policy, state, eps):
		action = policy(state).argmax(1)
		return torch.where(torch.rand_like(action, dtype=torch.float) < eps, torch.randint_like(action, self.num_actions), action)


	def train_encoder_decoder(self, replay_buffer, batch_size=32):
		state, action, next_state, reward, not_done = replay_buffer.sample()

		recons_next, recons_reward, lat = self.encoder_decoder(state, action)
		ed_loss = F.mse_loss(recons_next, next_state[:,0]) + 0.1 * F.mse_loss(recons_reward, reward) + 0.1 * lat.pow(2).mean()

		self.ed_optimizer.zero_grad()
		ed_loss.backward()
		self.ed_optimizer.step()


	def train_SR(self, replay_buffer, policy, batch_size=32):
		state, action, next_state, reward, not_done = replay_buffer.sample()

		with torch.no_grad():
			next_action = self.e_greedy(policy, next_state, self.eps)

			# Compute the target Q value
			target_sr = self.Q_target(next_state)
			target_sr = self.encoder_decoder.latent(state, action) + not_done * self.discount * target_sr[range(state.shape[0]), next_action.reshape(-1)]

		# Get current Q estimates
		current_sr = self.Q(state)[range(state.shape[0]), action.reshape(-1)]

		# Compute critic loss
		Q_loss = F.mse_loss(current_sr, target_sr)

		# Optimize the critic
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def train_OPE(self, replay_buffer, policy, batch_size=32):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		start_state = replay_buffer.start_sample(batch_size)
		
		with torch.no_grad():
			start_action = self.e_greedy(policy, start_state, self.eps)
			
			sr = self.Q(start_state)[range(start_state.shape[0]), start_action.reshape(-1)]
			start_Q = (1. - self.discount) * sr.mean(0)

			latent = self.encoder_decoder.latent(state, action)

		start_Q = (start_Q * self.w).mean()
		b_Q = (latent * self.w).mean(1).pow(2).mean()
		w_loss = 0.5 * b_Q - start_Q

		self.w_optimizer.zero_grad()
		w_loss.backward()
		self.w_optimizer.step()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())


	def eval_policy(self, replay_buffer, policy, batch_size=100):
		sum_val = 0
		for _ in range(100):
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
			sum_val += 1./(1. - self.discount) * float( ((self.w * self.encoder_decoder.latent(state, action)).mean(1,keepdim=True) * reward ).mean())
		
		return sum_val / 100.