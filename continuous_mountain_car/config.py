import torch

class Config:
    ENV_NAME = 'MountainCarContinuous-v0'
    STATE_DIMENSION = 2
    ACTION_DIMENSION = 1
    GAMMA = 0.99
    ACTION_BOUND = 2
    

    
    # torch
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    
    
    # Training parameters
    BATCH_SIZE = 128
    NUM_SAMPLES = 1024
    BUFFER_SIZE = 4096
    CRITIC_LR = 1e-3
    ACTOR_LR = 1e-3
    TAU = 0.995