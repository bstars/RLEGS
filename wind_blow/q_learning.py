import numpy as np
import matplotlib.pyplot as plt


import wind_env
from policy import epsilon_greedy_policy

def epsilon_greedy_policy(Q, state, action_space, eps):
    """
    Epsilon-greedy policy implicitly represented by Q function

    Args:
        Q (_type_): A look-up table that map (state, action) pair to Q value
        state (_type_): _description_
        action_space (_type_): _description_
        eps (float): epsilon in epsilon-greedy policy
    """
    A = len(action_space)
    probs = np.ones([A]) * eps / A
    Qs = Q[state, action_space]
    a = np.argmax(Qs)
    probs[a] += 1 - eps
    return np.random.choice(action_space, p=probs)

def off_policy_q_learning(epsilon=0.1, gamma=0.99, alpha=0.5):
    env = wind_env.WindEnv()
    action_space = np.arange(len(env.action_space))
    Q = np.zeros([len(env.state_space), len(env.action_space)])


    num_episodes = 0
    num_steps = []

    for _ in range(500):
        num_episodes += 1
        num_step = 0
        env.reset()

        eps = epsilon / num_episodes ** 2

        while not env.terminated:
            num_step += 1

            state = env.state
            a =  epsilon_greedy_policy(Q, state, action_space, eps)  
            action = env.action_space[a]  
            statep, reward, _ = env.step(action)
            ap = epsilon_greedy_policy(Q, statep, action_space, 0)

            delta = reward + gamma * Q[statep, ap] - Q[state, a]
            Q[state, a] += alpha * delta
        num_steps.append(num_step)
        
    return Q, num_steps


