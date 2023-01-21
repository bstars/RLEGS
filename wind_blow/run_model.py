
import numpy as np
import matplotlib.pyplot as plt

import wind_env
import sarsa
import q_learning




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


def plot_mat(ax, X):


    ax.matshow(X, cmap=plt.cm.Blues)

    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            c = X[j,i]
            ax.text(i, j, "%.1f"%c, va='center', ha='center')


def run_by_Q(Q, init=False):
    env = wind_env.WindEnv()
    env.reset()

    if init:
        env.state = env.coord_to_idx(np.array([3,0]))
    states = [env.state]

    while not env.terminated:
    # for _ in range(10):
        s = env.state

        a = np.argmax(Q[s,:])
        print(a)
        action = env.action_space[a]
        print( env.idx_to_coord(s), action)
        sp, _, _ = env.step(action)
        states.append(sp)
    states = [env.idx_to_coord(s) for s in states]
    return np.array(states)

def plot_board(ax):
    env = wind_env.WindEnv()
    n_plot = 100

    # plot horizontal lines
    xs = np.linspace(0, env.w, n_plot)
    for i in range(env.h + 1):
        ax.plot(xs, np.ones([n_plot]) * i, 'b-')

    # plot vertical lines
    ys = np.linspace(0, env.h, n_plot)
    for i in range(env.w + 1):
        ax.plot(np.ones([n_plot]) * i, ys, 'b-')



def plot_states(ax, states):
    env = wind_env.WindEnv()
    Is, Js = states[:,0], states[:,1]
    ax.plot(Js + 0.5, env.h - Is - 0.5, 'r-')
    

def plot_steps(ax, num_steps):
    acc = 0
    plots = []
    for i,n in enumerate(num_steps):
        l = [acc for _ in range(n)]
        acc += 1
        plots += l
    ax.plot(plots)
    ax.set_xlabel('time steps')
    ax.set_ylabel('episodes')
    
    
if __name__ == '__main__':
    # Q, num_steps = sarsa.on_policy_sarsa()
    # Q, num_steps = sarsa.on_policy_backward_sarsa()
    Q, num_steps = q_learning.off_policy_q_learning()
    V, A = QtoV(Q)
 


    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    plot_mat(ax1, V)
    plot_steps(ax2, num_steps)

    states = run_by_Q(Q, init=True)
    plot_board(ax3)
    plot_states(ax3,states)
    plt.show()

    

