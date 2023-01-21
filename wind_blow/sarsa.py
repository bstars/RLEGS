
import numpy as np
import matplotlib.pyplot as plt

import wind_env
from policy import epsilon_greedy_policy

def QtoV(Q):
    env = wind_env.WindEnv()
    Af = np.argmax(Q, axis=1)

    Vf = Q[np.arange(len(Af)), Af]

    V = np.zeros([env.h, env.w])
    A = np.zeros([env.h, env.w])
    for t in range(len(Vf)):
        i,j = env.idx_to_coord(t)
        V[i,j] = Vf[t]
        A[i,j] = Af[t]
    return V, A


def on_policy_sarsa(epsilon=0.1, gamma=0.99, alpha=0.5):
    """
    Solve the WindBlow model with Sarsa

    Args:
        epsilon (float): epsilon in \epsilon-greedy policy
        gamma (float): Discount factor
        alpha (float): Learning rate
    """
    env = wind_env.WindEnv()
    action_space = np.arange(len(env.action_space))

    Q = np.zeros([ len(env.state_space), len(env.action_space) ]) # Initialize the Q function

    num_episode = 0
    num_steps = []
    # while True:
    for _ in range(300):

        num_episode += 1
        num_step = 0
        env.reset()
        eps = epsilon * 1 / num_episode 


        state = env.state
        a = epsilon_greedy_policy(Q, state, action_space, eps)

        while not env.terminated:
            
            action = env.action_space[a]
            statep, reward, terminated = env.step(action)
            ap = epsilon_greedy_policy(Q, statep, action_space, eps)
            delta = reward + gamma * Q[statep, ap] - Q[state, a] # TD error

            Q[state, a] = Q[state, a] + alpha * delta 
            num_step += 1

            state = statep
            a = ap
        num_steps.append(num_step)

        # if np.linalg.norm(Q_copy - Q) <= 1e-7:
        #     return Q, num_steps
    return Q, num_steps

def on_policy_backward_sarsa(epsilon=0.1, gamma=0.99, alpha=0.5, lamb=0.5):
    """
    Solve the WindBlow model with Backward Sarsa by eligibility trace

    Args:
        epsilon (float, optional): epsilon in \epsilon-greedy policy. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        alpha (float, optional): Learning rate. Defaults to 0.5.
        lamb (float, optional): The weighting in Sarsa(lambda). Defaults to 0.5.
    """
    env = wind_env.WindEnv()
    action_space = np.arange(len(env.action_space))

    Q = np.zeros([ len(env.state_space), len(env.action_space) ])

    num_episodes = 0
    num_steps = []

    # while True:
    for _ in range(300):
        num_episodes += 1
        num_step = 0
        env.reset()
        eps = epsilon * 1 / num_episodes # GLIE policy

        # initialize eligibility trace
        E = np.zeros_like(Q)
        state = env.state
        a = epsilon_greedy_policy(Q, state, action_space, eps)

        while not env.terminated:
            # state = env.state
            E = gamma * lamb * E
            
            action = env.action_space[a]
            statep, reward, terminated = env.step(action)
            ap = epsilon_greedy_policy(Q, statep, action_space, eps)
            delta = reward + gamma * Q[statep, ap] - Q[state, a] # TD error
            E[state, a] += 1


            # for each state-action pair
            Q += alpha * delta * E
            # E = gamma * lamb * E

            a = ap
            state = statep

            num_step += 1
        num_steps.append(num_step)
    
    return Q, num_steps
        



            
        

def test():
    env = wind_env.WindEnv()
    Q = np.random.randn(len(env.state_space), len(env.action_space))
    action_space = np.arange(len(env.action_space))
    state = env.state


    a = epsilon_greedy_policy(Q, state, action_space, 0.05)
    print(Q[state])
    print(state, env.action_space[a])


if __name__ == '__main__':
    # test()
    Q, num_steps = on_policy_sarsa(0.1)
    acc = 0
    plots = []
    for i,n in enumerate(num_steps):
        l = [acc for _ in range(n)]
        acc += 1
        plots += l
    plt.plot(plots)
    plt.xlabel('time steps')
    plt.ylabel('episodes')
    plt.show()