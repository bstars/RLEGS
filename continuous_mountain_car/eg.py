import gym

env = gym.make(
    'MountainCarContinuous-v0'
).env
# observation_space = env.observation_space.shape[0]
# action_space = env.action_space.n

env.reset()

num_steps = 0
while True:
    # env.render()
    action = env.action_space.sample()
    state_next, reward, terminal, info  = env.step(action)
    # print(state_next, action, reward, terminal)
    # print()
    if terminal:
        break
    num_steps += 1
print(num_steps)
print(env.action_space)