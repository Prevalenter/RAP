import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# sa_mean = torch.tensor([9.9483e-01, 1.5116e-01, 9.9622e-01, 1.5213e-01, 1.2809e-04, 6.5528e-05], device='cuda:0') [None, :]
# sa_std = torch.tensor([0.0356, 0.0358, 0.0385, 0.0386, 0.0006, 0.0006], device='cuda:0')[None, :]
# print(sa_mean.shape, sa_std.shape)

# hole is [0.478, 0.397]
# +/- 0.02
state_max_min = torch.tensor([ [0.498, 0.417], [0.458, 0.377] ], device='cuda:0')[None]
action_max_min = torch.tensor( [ [0.0005, 0.0005], [-0.0005, -0.0005] ], device='cuda:0')[None]
print(state_max_min.shape, action_max_min.shape)
import numpy as np

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.bn1 = nn.BatchNorm1d(num_features=256)
		self.bn2 = nn.BatchNorm1d(num_features=256)
		self.max_action = max_action
		

	def forward(self, state):
		# 0-1 normlization
		state = (state-state_max_min[:, 1])/(state_max_min[:, 0]-state_max_min[:, 1])
		# print('actor state shape: ', state.shape, state.min(), state.max())

		# state = (state-sa_mean[:, :4])/sa_std[:, :4]
		# if state.shape[0]!=1: print(state.shape, state.mean(axis=0), state.std(axis=0))
		# print(state) 
		a = F.relu(self.l1(state))
		# print(a.mean(), a.std())
		a = self.bn1(a)
		a = F.relu(self.l2(a))
		# a = self.bn2(a)
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.bn1 = nn.BatchNorm1d(256)
		self.bn2 = nn.BatchNorm1d(256)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

		self.bn3 = nn.BatchNorm1d(256)
		self.bn4 = nn.BatchNorm1d(256)

	def forward(self, state, action):
		state = (state-state_max_min[:, 1])/(state_max_min[:, 0]-state_max_min[:, 1])
		action = (action - action_max_min[:, 1]) / (action_max_min[:, 0] - action_max_min[:, 1])
		# print('critic state shape: ', state.shape, state.min(), state.max())
		# print('critic action shape: ', action.shape, action.min(), action.max())

		sa = torch.cat([state, action], 1)

		# sa = (sa-sa_mean)/sa_std

		# if sa.shape[0]!=1: print(sa.shape, sa.mean(axis=0), sa.std(axis=0))

		q1 = F.relu(self.l1(sa))
		q1 = self.bn1(q1)
		q1 = F.relu(self.l2(q1))
		# q1 = self.bn2(q1)
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = self.bn3(q2)
		q2 = F.relu(self.l5(q2))
		# q2 = self.bn4(q2)
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		# print('Q1 state shape: ', state.shape)
		state = (state-state_max_min[:, 1])/(state_max_min[:, 0]-state_max_min[:, 1])
		action = (action - action_max_min[:, 1]) / (action_max_min[:, 0] - action_max_min[:, 1])

		# print('Q1 state shape: ', state.shape, state.min(), state.max())
		# print('Q1 action shape: ', action.shape, action.min(), action.max())

		sa = torch.cat([state, action], 1)

		# sa = (sa-sa_mean)/sa_std

		q1 = F.relu(self.l1(sa))
		q1 = self.bn1(q1)
		q1 = F.relu(self.l2(q1))
		# q1 = self.bn2(q1)
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		seed,
		file_name='file_name',
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.seed = seed
		self.file_name = file_name

		self.total_it = 0

		self.critic_loss_list = []
		self.actor_loss_list = []


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			# print('noise', noise.sum())

			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#torch.clip(critic_loss, max=0.5)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# print('critic_loss', critic_loss)
		self.critic_loss_list.append(critic_loss.item())
		self.actor_loss = []
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			# print('actor_loss', actor_loss)
			self.actor_loss_list.append(actor_loss.item())
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# if self.total_it%1e4:
		# 	np.save('loss/critic_loss_list_%s'%self.seed, self.critic_loss_list)
		# 	np.save('loss/actor_loss_list_%s'%self.seed, self.actor_loss_list)		

		if self.total_it%2e4:
			np.save('%s_critic_loss'%self.file_name, self.critic_loss_list)
			np.save('%s_actor_loss'%self.file_name, self.actor_loss_list)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		print('=====load=======')
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		