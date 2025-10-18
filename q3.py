import gym
import frogger_env
import pygame
import argparse
import time
import numpy as np
np.bool8 = np.bool_ # take care of incompatibility between gym==0.25.2 and numpy > 2.0
from gym.wrappers import RecordVideo
gym.logger.set_level(40) # suppress warnings on gym
from warnings import filterwarnings
filterwarnings(action="ignore", category=DeprecationWarning)
filterwarnings(action="ignore")

# Models and computation
import torch # will use pyTorch to handle NN 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from random import sample

# Visualization
import matplotlib
import matplotlib.pyplot as plt


''' 
    Simple test to verify that you can import the environment.  
    The agent samples always the up action. 
    Change the code below if you want to randonly sample an action, 
    or manually control the agent with the keyboard (w = up, s = down, a=left, d=right, other_key = idle)!
'''
env = gym.make("frogger-v1")
mode = "random" #change this to manual or random
if mode == 'manual':
    env.config["manual_control"] = True
env.config["observation"]["type"] = "lidar"


#The observation that the agent receives is a history of lidar scans + the distance to the goal + direction to the goal.
print('observation space:', env.observation_space)
#The action space consists of 5 actions 0: stand still, 1: move up 2: move down, 3: move left, 4: move right
print('action space:', env.action_space)

for _ in range(5):
    done = False
    action = 1
    obs = env.reset()
    total_reward = 0
    while not done:
        obs_, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        obs = obs_
        if env.config["manual_control"]:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                action = 1
            elif keys[pygame.K_s]:
                action = 2
            elif keys[pygame.K_a]:
                action = 3
            elif keys[pygame.K_d]:
                action = 4
            else:
                action = 0
        elif mode == 'up':
            action = 1
        elif mode == 'random':
            action = env.action_space.sample()
    print(total_reward)
env.close()


'''
    Question 3: Port your modified DDQN code from Question 2 and consider tuning 
    a non-trivial hyperparameter to improve the performance of your model in the 
    frogger-v1 environment. Please see the project descriprtion for more details.  
''' 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, filename):
    """
    Saves a model to a file 
        
    Parameters
    ----------
    model: your Q network
    filename: the name of the checkpoint file
    """
    torch.save(model.state_dict(), filename)


######################## Your code ####################################



