from random import sample
from matplotlib.pyplot import step
from sympy import Q, false, solve
import torch
from torch import nn 
import numpy as np
import gym
from copy import deepcopy
from time import time

from config import Config

global_env = gym.make('MountainCar-v0').env

class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dims = [Config.STATE_DIMENSION, 32, 64, 32, Config.NUM_ACTIONS]
        layers = []
        for i in range(len(dims)-2):
            layers.append( nn.Linear(dims[i], dims[i+1]) )
            layers.append( nn.ReLU() )
        
        layers.append( nn.Linear(dims[-2], dims[-1]) )
        self.layers = nn.Sequential(*layers)
        
        self.to(Config.DEVICE)
        
    def forward(self, x, actions=None, eps=None):
        """forward _summary_

        :param x: 
        :param actions: 
        :param eps: 
        """
        
        qs = self.layers(x)
        batch, action_dim = qs.shape
        
        
        if actions is not None:
            return qs[np.arange(batch), actions], actions
        
        actions = torch.argmax(qs, dim=1).detach().numpy()
        if eps is not None:
            # epsilon-greedy
            probs = np.zeros(qs.shape)
            probs[np.arange(batch), actions] = 1 - eps
            probs += eps / action_dim
            action_space = np.arange(action_dim)
            actions = np.array([ np.random.choice(action_space, p=probs[i]) for i in range(batch) ])
        return qs[np.arange(batch), actions], actions
    
    def run_model(self, max_steps=1000):
        # global_env = gym.make('MountainCar-v0').env
        state = global_env.reset()
        steps = 0
        for _ in range(max_steps):
            global_env.render()
            _, actions = self(
                torch.Tensor(state[None, :]).to(Config.DEVICE), eps=0
            )
            action = actions[0]
            state_next, reward, terminal, info = global_env.step(action)
            state = state_next
            steps += 1

            if terminal:
                return True, steps
        return False, steps


            
    
def array_to_tensor(arr):
    return torch.Tensor(arr).to(Config.DEVICE)


        
class Solver():
    def __init__(self) -> None:
        self.training_net = DQN()
        self.target_net = DQN()
        self.env = gym.make('MountainCar-v0').env
        self.env.reset()
        self.buffer = []
        self.episodes = 0
        
    def switch(self):
        
        target_dict = self.target_net.state_dict()
        self.target_net.load_state_dict(self.training_net.state_dict())
        self.training_net.load_state_dict(target_dict)
        
    def sample(self, random=False, eps=0.1):
        """sample

        Sample (s, a, r, s') pairs according 
        :param random: 
            If True, sample the action randomly
            If False, sample the action from epsilon-greedy policy according to training_net
        """
        action = self.env.action_space.sample()
        state, _, terminal, _ = self.env.step(action)
        ret = []
        
        while len(ret) < Config.NUM_SAMPLES:
            
            if random:
                action = self.env.action_space.sample()
            else:
                _, actions = self.training_net(
                    torch.Tensor(state[None, :]).to(Config.DEVICE), eps=eps
                )
                action = actions[0]
        
            state_next, reward, terminal, _ = self.env.step(action)
            ret.append((state, action, 10 if state_next[0] >= 0.5 else -1, state_next))
            
            if terminal:
                ret.append((state_next, self.env.action_space.sample(), 0, state_next))
                # ret.append((state_next, self.env.action_space.sample(), 0, state_next))
                state = self.env.reset()
                self.episodes += 1
            else:
                state = state_next
                
        self.buffer += ret
        
        if len(self.buffer) >= Config.BUFFER_SIZE:
            probs = np.ones([len(self.buffer)]) / len(self.buffer)
            sample_idx = np.random.choice(np.arange(len(self.buffer)), p=probs, size=Config.BUFFER_SIZE)
            self.buffer = [self.buffer[i] for i in sample_idx]
        np.random.shuffle(self.buffer)
        return ret
            
    def sample_from_buffer(self):
        probs = np.ones([len(self.buffer)]) / len(self.buffer)
        sample_idx = np.random.choice(np.arange(len(self.buffer)), p=probs, size=Config.BATCH_SIZE)
        batch = [self.buffer[i] for i in sample_idx]
        states = np.array([ r[0] for r in batch ])
        actions = np.array([ r[1] for r in batch ])
        rewards = np.array([ r[2] for r in batch ])
        states_next = np.array([ r[3] for r in batch ])
        return states, actions, rewards, states_next
        # return [self.buffer[i] for i in sample_idx]
    
    def train(self):
        self.sample(random=True)
        self.sample(random=True)
        loss = nn.MSELoss(reduction='mean')

        tic = time()
        iter = 0
        while True:
            iter += 1
            
            optimizer = torch.optim.Adam(params=self.training_net.parameters(), lr=Config.LEARNING_RATE)
            for _ in range(1000):
                
                states, actions, rewards, states_next = self.sample_from_buffer()
                
                with torch.no_grad():
                    # TD target with no grad
                    targets, _ = self.target_net(
                        array_to_tensor(states_next), eps=None
                    )
                    targets = array_to_tensor(rewards) + Config.GAMMA * targets
                
                predictions, _ = self.training_net(
                    array_to_tensor(states), actions
                )
                
                optimizer.zero_grad()
                l = loss(predictions, targets)
                l.backward()
                optimizer.step()
            print("%d iterations, loss=%.5f, %d episodes" % (iter, l.item(), self.episodes))
            
            
            if iter % 10 == 0:
                end, steps = self.training_net.run_model(max_steps=400)
                toc = time()
                elapsed = toc - tic
                print()
                if end:
                    print("After %f seconds, %d iterations, %d episodes, one episode in %d steps" % (elapsed, iter, self.episodes, steps))
                else:
                    print("After %f seconds, %d iterations, %d episodes" % (elapsed, iter, self.episodes))
                print()
            self.sample( random=False, eps=0.6 )
            self.switch()
            
            # if self.episodes >= 500:
                
            
            
                
        
if __name__ == '__main__':
    solver = Solver()
    solver.train()