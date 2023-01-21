import torch

class Config:
    STATE_DIMENSION = 2
    ACTION_DIMENSION = 1
    NUM_ACTIONS = 3
    GAMMA = 0.99
    
    
    
    # torch
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    
    
    # Training parameters
    BATCH_SIZE = 128
    NUM_SAMPLES = 1024
    BUFFER_SIZE = 4096
    LEARNING_RATE = 1e-4
    