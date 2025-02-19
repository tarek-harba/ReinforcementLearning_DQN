import gymnasium as gym, ale_py
import torch, torch.nn
from dqn import *
from train import *
import numpy as np

if __name__ == "__main__":
    ################ Initialize Environment ################
    env = gym.make(
        "ALE/Breakout-v5", full_action_space=True, obs_type="grayscale"
    )  # render_mode='human'

    ################ Train Agent in Environment ################
    parameters = {'n_input_frames' : 3,
    'minibatch_size' : 32,
    'epsilon' : float(1),
    'gamma' : float(0.6),
    'rem_size' : 1e6,
    "epsilon_decay" : float(0.9 / 1e6),
    'c': 1000}

    ################ Initialize Neural Network ################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = DQN(action_shape=env.action_space.n, n_input_frames=parameters['n_input_frames'])

    Q, epsilon_values, per_episode_reward, loss_values, episode_nr, frame_count= train(env=env, network=Q, parameters=parameters, device=device)

    ################ Save Model and  Performance Results ################
    torch.save(Q.state_dict(), f'{os.path.abspath(os.getcwd())}\Q_network.pth')
    np.savez("Q_perf.npz", epsilon_values=epsilon_values, loss_values=loss_values,
             per_episode_reward=per_episode_reward, frame_count=frame_count, episode_nr=episode_nr)