import gymnasium as gym, ale_py
import torch, random, torch.nn as nn, sys
from collections import deque, namedtuple
from torch import tensor
import os
import numpy as np
import copy

def sample_minibatch(n_input_frames, rem, minibatch_size, gamma, Q_hat, device):

    minibatch_actions = torch.zeros(minibatch_size).to(device)
    y = torch.zeros(minibatch_size).to(device)
    minibatch = []

    for i in range(minibatch_size):
        rand_idx = random.randint(n_input_frames - 1 , len(rem) - 1)
        next_sample = torch.cat(
            [
                rem[rand_idx].new_state,
                rem[rand_idx - 1].new_state,
                rem[rand_idx - 2].new_state,
            ],
            dim=1,
        ).to(device)

        current_sample = torch.cat(
            [
                rem[rand_idx].state,
                rem[rand_idx - 1].state,
                rem[rand_idx - 2].state,
            ],
            dim=1,
        ).to(device)
        minibatch.append(current_sample)
        r = rem[rand_idx].reward
        if rem[rand_idx].is_terminated:
            y[i] = r
        else:
            with torch.no_grad():
                y[i] = r + gamma * torch.max(Q_hat(next_sample)).item()

        minibatch_actions[i] = rem[rand_idx].action


    minibatch = torch.stack(minibatch).to(device) #TODO: ensure proper dimension of 32, 3, H, W
    minibatch = torch.squeeze(minibatch, 1)
    return minibatch, minibatch_actions, y



def train(env, network, parameters: dict, device):
    obs_space = env.observation_space
    action_space = env.action_space
    avg_pool = nn.AvgPool2d(kernel_size=2)

    ######## Hyperparameters ########
    n_input_frames = parameters['n_input_frames']
    minibatch_size = parameters['minibatch_size']
    epsilon = parameters['epsilon']
    gamma = parameters['gamma']
    epsilon_decay = parameters['epsilon_decay']
    c = parameters['c']

    ######## Training Specific  ########
    frame_count = 0
    episode_nr = 0
    epsilon_values = [] # epsilon value over episodes
    loss_values = [] # loss value over batch_size episodes
    per_episode_reward = [] #total reward over  batch_size episodes
    rem = deque(maxlen=int(parameters['rem_size']))
    sars = namedtuple(
        "sars", ["state", "action", "reward", "new_state", "is_terminated"]
    )  # SAR (t), ST (t+1),


    ######## Network Specific ########
    Q = network
    Q.to(device)
    Q_hat = copy.deepcopy(Q)
    Q_hat.to(device)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.RMSprop(
        params=Q.parameters(), lr=0.00025, momentum=0.95, eps=1e-6) #
    # For each episode out of many
    while frame_count < 10e6:
        in_episode_reward = []
        episode_nr += 1
        state, info = env.reset()
        terminated, truncated = False, False
        # For each step in trajectory / episode / roll-out
        while not terminated:  # or truncated

            ############ Sampling ############
            if frame_count < n_input_frames:
                action = action_space.sample()
            else:
                if random.uniform(0, 1) < epsilon:
                    action = action_space.sample()
                else:
                    recent_obs = torch.cat(
                        [
                            rem[-1].new_state,
                            rem[-2].new_state,
                            rem[-3].new_state,
                        ],
                        dim=1,
                    ).to(device)
                    with torch.no_grad():
                        action = torch.argmax(Q(recent_obs)).item()

            new_state, reward, terminated, truncated, info = env.step(action)
            rem.append(
                sars(
                    avg_pool(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)),
                    action,
                    reward,
                    avg_pool(
                        torch.from_numpy(new_state).float().unsqueeze(0).unsqueeze(0)),
                    terminated,))  # Append sars_t tuple


            frame_count += 1
            state = new_state

            ############ Training ############
            if frame_count > n_input_frames:  # if enough frames stored
                minibatch, minibatch_actions, y = sample_minibatch(n_input_frames, rem, minibatch_size, gamma, Q_hat, device)


                optimizer.zero_grad()
                minibatch_output = Q(minibatch)
                minibatch_actions = minibatch_actions.to(torch.int64)
                minibatch_output = minibatch_output.gather(1, minibatch_actions.unsqueeze(1))

                loss = loss_fn(input=minibatch_output, target=y.unsqueeze(1))
                loss_values.append(loss)
                loss.backward()
                optimizer.step()

            if reward > 0:
                in_episode_reward.append(reward)
            else:
                in_episode_reward.append(0)

            if frame_count < 1e6:  # decay epsilon for first 1M frames, then keep constant
                epsilon -= epsilon_decay

            epsilon_values.append(epsilon)
            if frame_count % c == 0:
                Q_hat.load_state_dict(Q.state_dict())  # Copy weights from Q to Q_hat
                Q_hat.to(device)

        per_episode_reward.append(sum(in_episode_reward))
        print(f'Episode {episode_nr}, Per Episode Reward: {per_episode_reward[-1]}, Epsilon: {epsilon}', f'Frame Count: {frame_count/1000}K')


    return Q, np.array(epsilon_values), np.array(per_episode_reward), np.array(loss_values), episode_nr, frame_count
