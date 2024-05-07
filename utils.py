import numpy as np
import matplotlib.pyplot as plt
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Optimal-Thinning-RL-PPO",

    # track hyperparameters and run metadata
    config={
        "memory_length": 0,
        "batch_size": 0,
        "num_epochs": 0,
        "learning_rate": 0,
        "epsilon": 0,
    }
)
