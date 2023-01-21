import gym
import numpy as np

env = gym.make('MountainCar-v0').env
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
env.reset()

num_steps = 0
while True:
    env.render()
    action = env.action_space.sample()
    state_next, reward, terminal, info = env.step(action)
    print(state_next, action, reward, terminal)
    if terminal:
        break
    num_steps += 1
print(num_steps)

# buffer = [1,2,3,4]
# probs = np.ones([len(buffer)]) / len(buffer)
# sample_idx = np.random.choice(np.arange(len(buffer)), p=probs, size=2)
# buffer = [buffer[i] for i in sample_idx]
# print(buffer)