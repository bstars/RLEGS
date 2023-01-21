
from math import fabs
import numpy as np


class WindEnv():

    def __init__(self) -> None:
        self.h = 7
        self.w = 10
        self.moveN = np.array([-1, 0])
        self.moveS = np.array([1, 0])
        self.moveW = np.array([0, -1])
        self.moveE = np.array([0, 1])

        self.action_space = [self.moveN, self.moveS, self.moveW, self.moveE]
        self.state_space = np.arange(0, self.h * self.w)
        self.terminate_state = np.array([3,7])
        

        self.reset()

    def reset(self):
        temp = self.coord_to_idx(self.terminate_state)
        state = np.random.choice(
            [i for i in range(temp-1)] + [i for i in range(temp + 1, len(self.state_space))]
        )
        self.state = state
        # self.state = self.coord_to_idx(np.array([3,0]))
        self.terminated = False

    def idx_to_coord(self, idx):
        return np.array([idx // self.w, idx % self.w])

    def coord_to_idx(self, coord):
        i, j = coord
        return i * self.w + j

    def step(self, a, return_pair = False):
        """

        Args:
            a (np.array): Should be in slf.action_space
            return_pair (bool, optional): Whether the return state is a pair or a index. Defaults to False.

        Returns:
            state (int or np.array): The terminated state, 
                                    if return_pair is True, the return is the coordinate
                                    if return_pair is False, the return is the state index
            reward (float):  The instant reward
            terminated (bool): Whether the resulting state is the terminate state
        """
        state = self.idx_to_coord(self.state)
        if np.all(state == self.terminate_state):
            state = state if return_pair else self.state
            self.terminated = True
            return state, 0, True
        
    
        endstate = state + a

        if state[1] in [3,4,5,8]:
            endstate[0] = endstate[0] - 1
        elif state[1] in [6,7]:
            endstate[0] = endstate[0] - 2
        state = endstate
        state[0] = max(state[0], 0)
        state[0] = min(state[0], self.h-1)

        state[1] = max(state[1], 0)
        state[1] = min(state[1], self.w-1)
        
        self.state = self.coord_to_idx(state)
        if np.all(state == self.terminate_state):  
            self.terminated = True
 
        if not return_pair: 
            state = self.coord_to_idx(state)
        return state, -1, self.terminated

     
def test_model():
    env = WindEnv()
    action_space = env.action_space
    i = 0
    while True:
        i += 1

        idx = np.random.choice(
            np.arange(len(action_space))
        )
        # print(env.idx_to_coord(env.state))
        a = action_space[idx]
        state, reward, terminated = env.step(a, return_pair=True)

        print(i, a, state, reward, terminated)
        print()
        if terminated:
            return 


if __name__ == '__main__':
    test_model()
    
    

