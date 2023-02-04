import numpy as np
import torch
from torch import nn
import gym

from time import time

from config import Config

global_env = gym.make(Config.ENV_NAME).env



def array_to_tensor(arr):
	return torch.Tensor(arr).to(Config.DEVICE)



class Critic(nn.Module):

	def __init__(self) -> None:
		super().__init__()
		dims = [Config.STATE_DIMENSION + Config.ACTION_DIMENSION, 32, 64, 32, 1]
		layers = []
		for i in range(len(dims) - 2):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(nn.ReLU())

		layers.append(nn.Linear(dims[-2], dims[-1]))
		self.layers = nn.Sequential(*layers)

		self.to(Config.DEVICE)

	def forward(self, s, a):
		sa = torch.cat([s, a], dim=1)
		return self.layers(sa)



class Actor(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		dims = [Config.STATE_DIMENSION, 32, 64, 32, Config.ACTION_DIMENSION]
		layers = []
		for i in range(len(dims) - 2):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(nn.ReLU())

		layers.append(nn.Linear(dims[-2], dims[-1]))
		layers.append(nn.Tanh())
		self.layers = nn.Sequential(*layers)

		self.to(Config.DEVICE)

	def forward(self, s, noise_scale=None):
		a = self.layers(s) * Config.ACTION_BOUND
		if noise_scale is None:
			return a

		m, n = a.shape
		noise = torch.randn([m, n]).to(Config.DEVICE) * noise_scale
		a = a + noise
		a = torch.clip(a, -1. * Config.ACTION_BOUND, 1. * Config.ACTION_BOUND)
		return a

	def run_model(self, max_steps=1000):
		state = global_env.reset()
		steps = 0
		for _ in range(max_steps):
			global_env.render()
			action = self(array_to_tensor(state[None, :]), noise_scale=None).cpu().detach().numpy()[0]
			state_next, _, terminal, _ = global_env.step(action)
			state = state_next
			steps += 1

			if terminal:
				return True, steps
		return False, steps

class DDPG():
	def __init__(self) -> None:
		self.critic = Critic()
		self.critic_target = Critic()
		self.actor = Actor()
		self.actor_target = Actor()
		self.env = gym.make(Config.ENV_NAME).env
		self.state = self.env.reset()
		self.buffer = []
		self.episodes = 0
		self.total_reward = 0
		self.total_reward_prev = 0
		self.steps = 0

		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(
				param.data
			)

		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(
				param.data
			)

	def soft_update(self):
		with torch.no_grad():
			for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
				param_target.data.mul_(Config.TAU)
				param_target.data.add_( (1 - Config.TAU) * param.data )

			for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
				param_target.data.mul_(Config.TAU)
				param_target.data.add_( (1 - Config.TAU) * param.data )


	def sample(self, batch_size=Config.BATCH_SIZE, random=False, noise_scale=0.1):
		"""sample _summary_

		:param random: _description_, defaults to False
		:param noise_scale: _description_, defaults to 0.1
		"""
		ret = []
		while len(ret) < batch_size:
			if random:
				action = self.env.action_space.sample()
			else:
				action = self.actor(
					torch.Tensor(self.state[None, :]).to(Config.DEVICE), noise_scale=noise_scale
				)
				action = action.cpu().detach().numpy()[0]

			state_next, reward, terminal, _ = self.env.step(action)
			ret.append((self.state, action, reward, state_next, 1 if terminal else 0))

			self.state = state_next
			self.total_reward += (Config.GAMMA ** self.steps) * reward
			self.steps += 1

			if terminal:
				ret.append((state_next, self.env.action_space.sample(), 0, state_next, True))
				self.total_reward_prev = self.total_reward
				self.total_reward = 0
				self.steps = 0
				self.episodes += 1
				self.state = self.env.reset()

		self.buffer += ret
		if len(self.buffer) >= Config.BUFFER_SIZE:
			self.buffer = self.buffer[-Config.BUFFER_SIZE:]
			# probs = np.ones([len(self.buffer)]) / len(self.buffer)
			# sample_idx = np.random.choice(np.arange(len(self.buffer)), p=probs, size=Config.BUFFER_SIZE)
			# self.buffer = [self.buffer[i] for i in sample_idx]
			# np.random.shuffle(self.buffer)
		return ret

	def sample_from_buffer(self):
		probs = np.ones([len(self.buffer)]) / len(self.buffer)
		sample_idx = np.random.choice(np.arange(len(self.buffer)), p=probs, size=Config.BATCH_SIZE)
		batch = [self.buffer[i] for i in sample_idx]
		states = np.array([r[0] for r in batch])
		actions = np.array([r[1] for r in batch])
		rewards = np.array([r[2] for r in batch])
		states_next = np.array([r[3] for r in batch])
		terminals = np.array([r[4] for r in batch])
		return array_to_tensor(states), \
			array_to_tensor(actions), \
			array_to_tensor(rewards), \
			array_to_tensor(states_next), \
			array_to_tensor(terminals)

	def train(self):
		self.sample(random=True, batch_size=Config.BATCH_SIZE * 2)

		qloss = nn.MSELoss(reduction='mean')
		# qloss = nn.L1Loss(reduction='mean')

		tic = time()
		iter = 0

		critic_optim = torch.optim.Adam(params=self.critic.parameters(), lr=Config.CRITIC_LR)
		actor_optim = torch.optim.Adam(params=self.actor.parameters(), lr=Config.ACTOR_LR)

		while True:

			iter += 1
			noise_scale = max(1 - (1e-7) * iter, 0.1)
			self.sample(random=False, noise_scale=noise_scale, batch_size=1)

			for _ in range(1):
				states, actions, rewards, states_next, terminals = self.sample_from_buffer()
				with torch.no_grad():
					actions_next = self.actor_target(states_next, noise_scale=None)
					targets = rewards[:,None] + Config.GAMMA * (1 - terminals[:,None]) * self.critic_target(states_next, actions_next)


				# critic training
				critic_optim.zero_grad()
				pred = self.critic(states, actions)

				closs = qloss(targets, pred)
				closs.backward()
				critic_optim.step()


				# Actor train
				actor_optim.zero_grad()
				scores = self.critic(states, self.actor(states))

				aloss = -torch.mean(scores)
				aloss.backward()
				actor_optim.step()

				# self.switch()
				self.soft_update()

			if iter % 10000 == 0:
				print('closs:%.4f, aloss:%.4f' % (closs.item(), aloss.item()))
				toc = time()
				print('%.5f seconds' % (toc - tic))
				end, steps = self.actor.run_model(max_steps=400)
				if end:
					print("\t %d iterations, %d episodes, one episode in %d steps, last episodic reward %.5f," % (
					iter, self.episodes, steps, self.total_reward_prev))
				else:
					print("\t %d iterations, %d episodes, last episodic reward %.5f" % (
					iter, self.episodes, self.total_reward_prev))
				print()

				# if iter % 5000 == 0:
				# 	torch.save(
				# 		{
				# 			"actor": self.actor.state_dict(),
				# 			"critic": self.critic.state_dict(),
				# 		},
				# 		'./iter_%d.pth' % iter
				# 	)

if __name__ == '__main__':
	solver = DDPG()
	solver.train()