import gym
import frogger_env
import pygame
import argparse
import time
import numpy as np
import os
import json
from collections import namedtuple
np.bool8 = np.bool_ # take care of incompatibility between gym==0.25.2 and numpy > 2.0
from gym.wrappers import RecordVideo
gym.logger.set_level(40) # suppress warnings on gym
from warnings import filterwarnings
filterwarnings(action="ignore", category=DeprecationWarning)
filterwarnings(action="ignore")

# Models and computation
import torch 
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
    Simple test for frogger-v1. 
    (w = up, s = down, a=left, d=right, other_key = idle)!
'''
env = gym.make("frogger-v1")
mode = "random" #change this to manual or random
if mode == 'manual':
    env.config["manual_control"] = True
env.config["observation"]["type"] = "lidar"


#The observation that the agent receives is a history of lidar scans + the distance to the goal + direction to the goal.
print('observation space:', env.observation_space, flush=True)
#The action space consists of 5 actions 0: stand still, 1: move up 2: move down, 3: move left, 4: move right
print('action space:', env.action_space, flush=True)

for _ in range(1): # Reduced for faster script startup
    done = False
    action = 1
    obs = env.reset()
    total_reward = 0
    while not done:
        obs_, reward, done, _ = env.step(action)
        total_reward += reward
        # env.render() # Commented out for non-interactive run
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
    print(f"Test episode reward: {total_reward}", flush=True)
env.close()


'''
    Question 3: Port your modified DDQN code from Question 2 and consider tuning 
    a non-trivial hyperparameter to improve the performance of your model in the 
    frogger-v1 environment. Please see the project descriprtion for more details.  
''' 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

def save_checkpoint(model, filename):
    """
    Saves a model to a file 
        
    Parameters
    ----------
    model: your Q network
    filename: the name of the checkpoint file
    """
    print(f"Saving model to {filename}...", flush=True)
    torch.save(model.state_dict(), filename)


######################## Your code ####################################

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

class DuelingQNetwork(nn.Module):
    """Neural Network for Dueling Q-values."""

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
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Shared layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Value stream
        self.value_fc1 = nn.Linear(fc1_units, fc2_units)
        self.value_fc2 = nn.Linear(fc2_units, 1) # Outputs a single value V(s)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(fc1_units, fc2_units)
        self.advantage_fc2 = nn.Linear(fc2_units, action_size) # Outputs one advantage A(s,a) for each action

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))

        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value) # V(s)

        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage) # A(s,a)

        # Combine V(s) and A(s,a) to get Q(s,a)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

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

    def __init__(self, state_size, action_size, seed, hyperparameters, ddqn=False, dueling=False):
        """Initialize an Agent object.
        
        Parameters
        ----------
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hyperparameters (dict): dictionary of hyperparameters
            ddqn (bool): flag to use Double DQN
            dueling (bool): flag to use Dueling Q-Network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.dueling = dueling
        
        # Hyperparameters
        self.buffer_size = hyperparameters["BUFFER_SIZE"]
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.gamma = hyperparameters["GAMMA"]
        self.tau = hyperparameters["TAU"]
        self.lr = hyperparameters["LR"] # <-- This is the one we'll sweep
        self.update_every = hyperparameters["UPDATE_EVERY"]

        # Q-Network
        if self.dueling:
            # print("Using Dueling Q-Network")
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            # print("Using Standard Q-Network")
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

        if self.ddqn:
            # --- Double DQN (DDQN) Target ---
            # 1. Get the best action from the LOCAL network
            best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            # 2. Get the Q-value for that action from the TARGET network
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        else:
            # --- Vanilla DQN Target ---
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
    eval_env = env
    if record_video:
        # Make sure the video folder exists
        os.makedirs(video_folder, exist_ok=True)
        # Record only the first episode of the evaluation run
        eval_env = RecordVideo(env, video_folder, episode_trigger=lambda x: x == 0)

    scores = []
    for i in range(n_episodes):
        state = eval_env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps=0.0) # Always use greedy policy for evaluation
            state, reward, done, _ = eval_env.step(action)
            score += reward
            if done:
                break
        scores.append(score)

    if record_video:
        eval_env.close()
        
    return np.mean(scores)

def train_dqn(env, agent, seed, model_name="dqn", n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, 
              eps_decay=0.995, eval_every=10000):
    """Deep Q-Learning training and evaluation loop.
    
    Parameters
    ----------
        model_name (str): Name for saving the checkpoint
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
    
    # Track the best evaluation score for saving the best model
    best_eval_score = -np.inf

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
                avg_eval_score = evaluate(agent, env)
                eval_scores.append(avg_eval_score)
                eval_steps.append(total_steps)
                print(f"\nSteps: {total_steps}\tEval Avg Score: {avg_eval_score:.2f}", flush=True)
                
                # Check if this is the best model so far
                if avg_eval_score > best_eval_score:
                    best_eval_score = avg_eval_score
                    # Save the best model checkpoint
                    save_checkpoint(agent.qnetwork_local, f"./models/{model_name}_checkpoint_{seed}.pth")

                next_eval_step += eval_every

            if done:
                break 
        
        training_scores_window.append(score)
        training_scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print(f'\rEpisode {i_episode}\tTraining Avg Score: {np.mean(training_scores_window):.2f}', end="", flush=True)
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tTraining Avg Score: {np.mean(training_scores_window):.2f}', flush=True)
    
    # Ensure at least one eval score is present, even if eval_every is large
    if not eval_scores:
        avg_eval_score = evaluate(agent, env)
        eval_scores.append(avg_eval_score)
        eval_steps.append(total_steps)
        print(f"\nFinal Eval Score: {avg_eval_score:.2f}", flush=True)
        if avg_eval_score > best_eval_score:
            best_eval_score = avg_eval_score
            save_checkpoint(agent.qnetwork_local, f"./models/{model_name}_checkpoint_{seed}.pth")

    return training_scores, eval_scores, eval_steps, best_eval_score

# ### MODIFIED MAIN BLOCK FOR Q3 ###
if __name__ == '__main__':
    
    # ------------------------------------------------------------------
    # ### SET MODE HERE ###
    # "train": Runs full training, saves models, and saves plot data.
    # "plot_and_video": Loads saved plot data and best model, skips training.
    MODE = "plot_and_video" 
    # ------------------------------------------------------------------

    # Define file paths
    PLOT_DATA_FILE = "./figures/q3_plot_data.npz"
    BEST_MODEL_INFO_FILE = "./models/q3_best_model_info.json" # For video

    # --- Q3: Define Hyperparameters to sweep ---
    # We will sweep the Learning Rate (LR)
    hyperparameter_sweep = [
        {"LR": 5e-4}, # Faster
        {"LR": 2e-4}, # original value from Q2
        {"LR": 1e-4}, # Slower
        {"LR": 5e-5}  # Much slower
    ]
    
    # This is the hyperparameter we are NOT sweeping
    base_hyperparameters = {
        "BUFFER_SIZE": int(1e5),
        "BATCH_SIZE": 64,
        "GAMMA": 0.99,
        "TAU": 1e-3,
        "UPDATE_EVERY": 4
    }
    
    # This is the model config we are using (best from Q2)
    model_config = {"name": "Dueling_DDQN", "ddqn": True, "dueling": True}

    # --- Multi-Seed Training ---
    seeds = [42] # Best seed for Duelling DDQN from previous question
    
    # Create directories for models, figures, and videos
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./videos", exist_ok=True)

    # Initialize containers for results
    processed_model_results = {}
    eval_steps_axis = None
    best_model_info_for_video = {}

    if MODE == "train":
        print("===== STARTING Q3 TRAINING MODE (Hyperparameter Sweep) =====", flush=True)
        
        # Store raw results before processing
        raw_model_results = {}
        global_min_len = np.inf

        # Track bests
        best_score_per_lr = {f"LR_{config['LR']}": -np.inf for config in hyperparameter_sweep}
        best_seed_per_lr = {f"LR_{config['LR']}": None for config in hyperparameter_sweep}
        global_best_score = -np.inf

        # --- Loop over the hyperparameter settings ---
        for sweep_config in hyperparameter_sweep:
            
            # Combine base hypers with the one being swept
            current_hyperparameters = {**base_hyperparameters, **sweep_config}
            lr_value = current_hyperparameters["LR"]
            
            # Create a unique name for this run
            run_name = f"{model_config['name']}_LR_{lr_value}"
            print(f"\n--- Training Run: {run_name} ---", flush=True)
            all_eval_scores_raw = []

            for seed in seeds:
                print(f"--- Training with Seed: {seed} ---", flush=True)
                
                # Set seeds for reproducibility
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # --- Use frogger-v1 ---
                env = gym.make("frogger-v1")
                env.config["observation"]["type"] = "lidar"
                env.seed(seed)
                
                state_size = env.observation_space.shape[0]
                action_size = env.action_space.n
                
                # --- Pass the new flags ---
                agent = Agent(state_size=state_size, action_size=action_size, seed=seed, 
                              hyperparameters=current_hyperparameters, # Pass the combined dict
                              ddqn=model_config["ddqn"], 
                              dueling=model_config["dueling"])

                # Train the agent
                _, eval_scores, current_eval_steps, best_score_for_seed = train_dqn(
                    env, agent, seed, model_name=run_name, n_episodes=10000, eval_every=10000
                )

                # --- Track bests ---
                lr_key = f"LR_{lr_value}"
                # Check if this is the best run for this LEARNING RATE
                if best_score_for_seed > best_score_per_lr[lr_key]:
                    best_score_per_lr[lr_key] = best_score_for_seed
                    best_seed_per_lr[lr_key] = seed

                # Check if this is the best run OVERALL (for video)
                if best_score_for_seed > global_best_score:
                    global_best_score = best_score_for_seed
                    best_model_info_for_video = {
                        "config": model_config, 
                        "hyperparameters": current_hyperparameters, 
                        "seed": seed, 
                        "score": best_score_for_seed
                    }
                
                all_eval_scores_raw.append(eval_scores)
                if eval_steps_axis is None:
                    eval_steps_axis = current_eval_steps
                
                # Find the global minimum run length
                if eval_scores: # Avoid error if a run fails and has no scores
                    global_min_len = min(global_min_len, len(eval_scores))
                
                env.close()

            # Store raw scores
            raw_model_results[run_name] = all_eval_scores_raw

        print("\n===== Q3 TRAINING COMPLETE. PROCESSING RESULTS... =====", flush=True)
        
        # --- Process all results AFTER training, using global_min_len ---
        eval_steps_axis = eval_steps_axis[:int(global_min_len)]
        save_plot_data = {'eval_steps_axis': eval_steps_axis}

        for run_name, raw_scores_list in raw_model_results.items():
            # Truncate all score lists to the global minimum length
            eval_scores_np = np.array([s[:int(global_min_len)] for s in raw_scores_list if s]) # check 'if s'
            
            if eval_scores_np.size == 0:
                print(f"Warning: No scores found for run {run_name}. Skipping.", flush=True)
                continue

            # Calculate mean and std
            mean_scores = np.mean(eval_scores_np, axis=0)
            std_scores = np.std(eval_scores_np, axis=0)
            
            # Store for plotting
            processed_model_results[run_name] = (mean_scores, std_scores)
            
            # Store for saving to file (mean, std)
            save_plot_data[run_name] = (mean_scores, std_scores)

        # --- Save plot data to file ---
        np.savez(PLOT_DATA_FILE, **save_plot_data)
        print(f"Plot data saved to {PLOT_DATA_FILE}", flush=True)

        # --- Save best model info to file (for video) ---
        with open(BEST_MODEL_INFO_FILE, 'w') as f:
            json.dump(best_model_info_for_video, f, indent=4)
        print(f"Overall best model info (for video) saved to {BEST_MODEL_INFO_FILE}", flush=True)

        # --- Print info for model submission ---
        print("\n===== Best Models for Submission (Q3) =====", flush=True)
        for lr_key, best_seed in best_seed_per_lr.items():
            if best_seed is not None:
                lr_value = lr_key.split('_')[1] # Get '5e-4' etc. from 'LR_5e-4'
                score = best_score_per_lr[lr_key]
                run_name = f"{model_config['name']}_LR_{lr_value}"
                print(f"  > Best for '{lr_key}': Seed {best_seed} (Score: {score:.2f})", flush=True)
                print(f"    - Submit file: ./models/{run_name}_checkpoint_{best_seed}.pth", flush=True)
            else:
                print(f"  > No successful runs for '{lr_key}'.", flush=True)

    elif MODE == "plot_and_video":
        print(f"===== LOADING Q3 DATA FOR PLOTTING & VIDEO =====", flush=True)
        
        # --- Load plot data ---
        try:
            data = np.load(PLOT_DATA_FILE, allow_pickle=True)
            eval_steps_axis = data['eval_steps_axis']
            for k in data.files:
                if k != 'eval_steps_axis':
                    # Data is saved as (mean_array, std_array)
                    processed_model_results[k] = (data[k][0], data[k][1])
            print(f"Loaded plot data from {PLOT_DATA_FILE}", flush=True)
        except FileNotFoundError:
            print(f"ERROR: Plot data file not found: {PLOT_DATA_FILE}", flush=True)
            print("Please run with MODE = 'train' first.", flush=True)
            exit()

        # --- Load best model info ---
        try:
            with open(BEST_MODEL_INFO_FILE, 'r') as f:
                best_model_info_for_video = json.load(f)
            print(f"Loaded best model info from {BEST_MODEL_INFO_FILE}", flush=True)
        except FileNotFoundError:
            print(f"ERROR: Best model info file not found: {BEST_MODEL_INFO_FILE}", flush=True)
            print("Please run with MODE = 'train' first.", flush=True)
            exit()
    
    else:
        print(f"ERROR: Unknown MODE '{MODE}'. Please set to 'train' or 'plot_and_video'.", flush=True)
        exit()


    # --- Plot Comparison Results ---
    if processed_model_results:
        print("Generating Q3 comparison plot...", flush=True)
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        # Sort keys to make the legend look nice (e.g., by LR value)
        sorted_keys = sorted(processed_model_results.keys(), key=lambda x: float(x.split('_LR_')[-1]))

        for run_name in sorted_keys:
            mean_scores, std_scores = processed_model_results[run_name]
            lr_label = run_name.split('Dueling_DDQN_')[-1] # Gets "LR_5e-05"
            
            # Check dimensions one last time
            if len(eval_steps_axis) != len(mean_scores):
                print(f"WARNING: Mismatch for {run_name}. X: {len(eval_steps_axis)}, Y: {len(mean_scores)}", flush=True)
                # Truncate mean/std to match x-axis
                min_len = min(len(eval_steps_axis), len(mean_scores))
                ax.plot(eval_steps_axis[:min_len], mean_scores[:min_len], label=lr_label)
                ax.fill_between(eval_steps_axis[:min_len], std_scores[:min_len], std_scores[:min_len], alpha=0.2)
            else:
                ax.plot(eval_steps_axis, mean_scores, label=lr_label)
                ax.fill_between(eval_steps_axis, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
        
        ax.set_ylabel('Average Score', fontsize=14)
        ax.set_xlabel('Training Steps', fontsize=14)
        ax.set_title('Dueling-DDQN Performance vs. Learning Rate (frogger-v1)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        plot_filename = './figures/q3_lr_sweep_evaluation.png'
        plt.savefig(plot_filename)
        print(f"Comparison plot saved to {plot_filename}", flush=True)
        # plt.show()
    else:
        print("No results to plot.", flush=True)

    # --- Final Evaluation and Video Recording (Uses OVERALL best from this sweep) ---
    if best_model_info_for_video:
        best_model_config = best_model_info_for_video.get('config')
        best_hyperparameters = best_model_info_for_video.get('hyperparameters')
        best_seed = best_model_info_for_video.get('seed')

        if not all([best_model_config, best_hyperparameters, best_seed is not None]):
            print("Error: Best model info file is incomplete. Skipping video.", flush=True)
        else:
            lr_val = best_hyperparameters['LR']
            run_name = f"{best_model_config['name']}_LR_{lr_val}"
            print(f"\n--- Recording final performance of OVERALL best model: {run_name} (Seed: {best_seed}) ---", flush=True)
            
            # Re-initialize the environment and agent with the best seed/config
            env = gym.make("frogger-v1")
            env.config["observation"]["type"] = "lidar"
            env.seed(best_seed)
            
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            # Create a new agent and load the saved weights
            final_agent = Agent(state_size=state_size, action_size=action_size, seed=best_seed, 
                                hyperparameters=best_hyperparameters,
                                ddqn=best_model_config["ddqn"],
                                dueling=best_model_config["dueling"])
            
            model_path = f"./models/{run_name}_checkpoint_{best_seed}.pth"
            
            try:
                print(f"Loading best model from: {model_path}", flush=True)
                final_agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))

                # Evaluate and record the video
                video_path = f'./videos/final_q3_best_{run_name}'
                final_score = evaluate(final_agent, env, n_episodes=5, record_video=True, video_folder=video_path)
                
                print(f"Final best agent achieved an average score of: {final_score:.2f}", flush=True)
                print(f"Video saved to {video_path}", flush=True)
            
            except FileNotFoundError:
                print(f"ERROR: Could not load model file: {model_path}", flush=True)
                print("Cannot generate final video.", flush=True)
            
            env.close()
    else:
        print("No best model was determined. Skipping video recording.", flush=True)

print("\nQ3 Script finished.", flush=True)