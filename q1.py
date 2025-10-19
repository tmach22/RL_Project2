import gym
import frogger_env
import pygame
import argparse
import time
import numpy as np
from collections import namedtuple
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
    or manually control the agent with the keyboard (w = up, s = down, other_key = idle)!

'''
print(gym.__version__)
env = gym.make("frogger-v0")
mode = "up" #change this to manual or random
if mode == 'manual':
    env.config["manual_control"] = True
env.config["observation"]["type"] = "lidar"
# env.config["observation"]["type"] = "occupancy_grid"


#The observation that the agent receives is a history of lidar scans + the distance to the goal.
print('observation space:', env.observation_space)
#The action space consists of three actions 0: stand still, 1: move up 2: move down
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
            else:
                action = 0
        elif mode == 'up':
            action = 1
        elif mode == 'random':
            action = env.action_space.sample()
    print(total_reward)
env.close()



'''
    Question 1: Implement DQN to solve the frogger-v0 environment and enable the agent 
    to reach the other side of the highway.You can work here or an a separate file. 
    Please see the project description for more details.   
''' 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_checkpoint(model, filename):
    """
    Saves a model to a file 
        
    Parameters
    ----------
    model: your Q network
    filename: the name of the checkpoint file
    """
    torch.save(model.state_dict(), filename)

class QNetwork(nn.Module):
    """Neural Network for approximating Q-values."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Parameters
        ----------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Parameters
        ----------
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hyperparameters):
        """Initialize an Agent object.
        
        Parameters
        ----------
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hyperparameters (dict): dictionary of hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Hyperparameters
        self.buffer_size = hyperparameters["BUFFER_SIZE"]
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.gamma = hyperparameters["GAMMA"]
        self.tau = hyperparameters["TAU"]
        self.lr = hyperparameters["LR"]
        self.update_every = hyperparameters["UPDATE_EVERY"]

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Parameters
        ----------
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Parameters
        ----------
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

def evaluate(agent, env, n_episodes=10, max_t=1000, record_video=False, video_folder='./videos'):
    """
    Evaluates a DQN agent's performance and optionally records a video.
    """
    # Use a separate, wrapped environment for video recording
    eval_env = env
    if record_video:
        # Record only the first episode of the evaluation run
        eval_env = RecordVideo(env, video_folder, episode_trigger=lambda x: x == 0)

    scores = []
    for i in range(n_episodes):
        # Use the potentially wrapped environment
        state = eval_env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps=0.0)
            state, reward, done, _ = eval_env.step(action)
            score += reward
            if done:
                break
        scores.append(score)

    # Be sure to close the wrapped environment to save the video
    if record_video:
        eval_env.close()
        
    return np.mean(scores)

def train_dqn(env, agent, seed, n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, 
              eps_decay=0.995, eval_every=5000):
    """Deep Q-Learning training and evaluation loop.
    
    Parameters
    ----------
        eval_every (int): Run evaluation every 'eval_every' steps.
    """
    training_scores = []
    training_scores_window = deque(maxlen=100)
    
    # Evaluation scores
    eval_scores = []
    eval_steps = []
    
    eps = eps_start
    total_steps = 0
    next_eval_step = eval_every

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            total_steps += 1

            # --- Evaluation Step ---
            if total_steps >= next_eval_step:
                # Run evaluation and store the results
                avg_eval_score = evaluate(agent, env)
                eval_scores.append(avg_eval_score)
                eval_steps.append(total_steps)
                print(f"\nSteps: {total_steps}\tEval Avg Score: {avg_eval_score:.2f}")
                next_eval_step += eval_every

            if done:
                break 
        
        training_scores_window.append(score)
        training_scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print(f'\rEpisode {i_episode}\tTraining Avg Score: {np.mean(training_scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tTraining Avg Score: {np.mean(training_scores_window):.2f}')
    
    # Save the trained model
    save_checkpoint(agent.qnetwork_local, f"dqn_checkpoint_{seed}.pth")
    return training_scores, eval_scores, eval_steps

# --- Main execution ---
if __name__ == '__main__':
    # Define Hyperparameters
    hyperparameters = {
        "BUFFER_SIZE": int(1e5),
        "BATCH_SIZE": 64,
        "GAMMA": 0.99,
        "TAU": 1e-3,
        "LR": 2e-4,
        "UPDATE_EVERY": 4
    }

    # --- Multi-Seed Training ---
    seeds = [0, 42, 123] # At least 3 different seeds
    all_eval_scores = []
    eval_steps = None # To store the steps axis

    global_best_score = -np.inf
    best_seed = None

    for seed in seeds:
        print(f"--- Training with Seed: {seed} ---")
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Setup environment and agent
        env = gym.make("frogger-v0")
        env.config["observation"]["type"] = "lidar"
        env.seed(seed)
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = Agent(state_size=state_size, action_size=action_size, seed=seed, hyperparameters=hyperparameters)

        # Train the agent
        _, eval_scores, current_eval_steps = train_dqn(env, agent, seed)

        if eval_scores:
            current_max_score = np.max(eval_scores)
            if current_max_score > global_best_score:
                global_best_score = current_max_score
                best_seed = seed
        
        all_eval_scores.append(eval_scores)
        if eval_steps is None:
            eval_steps = current_eval_steps
        
        env.close()

    # --- Process and Plot Results ---
    if all_eval_scores:
        # Find the minimum length among all score lists
        min_len = min(len(s) for s in all_eval_scores)

        # Truncate all lists to that minimum length
        eval_scores_np = np.array([s[:min_len] for s in all_eval_scores])
        if eval_steps:
            eval_steps = eval_steps[:min_len]

        mean_scores = np.mean(eval_scores_np, axis=0)
        std_scores = np.std(eval_scores_np, axis=0)

        # Plot the mean evaluation scores with std deviation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(eval_steps, mean_scores, label='Mean Eval Score')
        plt.fill_between(eval_steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.3, label='Std Dev')
        plt.ylabel('Average Score')
        plt.xlabel('Training Steps')
        plt.title('DQN Evaluation Performance')
        plt.legend()
        plt.savefig('/data/nas-gpu/wang/tmach007/RL_Project2/figure/dqn_evaluation_performance.png')
    else:
        print("No evaluation scores were recorded.")
        # Exit or handle the case where there's no data to plot

    # --- Final Evaluation and Video Recording ---
    print("\n--- Recording final agent performance... ---")
    
    # Re-initialize the environment and agent with the best seed
    final_seed = seeds[best_seed] # or whichever seed performed best
    env = gym.make("frogger-v0")
    env.config["observation"]["type"] = "lidar"
    env.seed(final_seed)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create a new agent and load the saved weights
    final_agent = Agent(state_size=state_size, action_size=action_size, seed=final_seed, hyperparameters=hyperparameters)
    print(f"Loading best model from seed: {best_seed}")
    final_agent.qnetwork_local.load_state_dict(torch.load(f'dqn_checkpoint_{best_seed}.pth'))

    # Evaluate and record the video
    final_score = evaluate(final_agent, env, n_episodes=5, record_video=True, video_folder='./videos/final_dqn')
    
    print(f"Final agent achieved an average score of: {final_score:.2f}")
    
    env.close()