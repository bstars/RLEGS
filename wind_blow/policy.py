import numpy as np

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

