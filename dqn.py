import torch.nn as nn, torch
import torch.nn.functional as F


class DQN(nn.Module):
    # Assumes: 3 recent frames as channels with image of size:[210,160]
    def __init__(self, action_shape, n_input_frames):
        super().__init__()

        self.h1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.h2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.h3 = nn.Sequential(
            nn.Linear(in_features=2816, out_features=256), nn.ReLU()
        )
        self.output = nn.Linear(in_features=256, out_features=action_shape)

    def forward(self, x):
        y = self.h1(x)
        y = self.h2(y)
        y = torch.flatten(y, start_dim=1)
        y = self.h3(y)
        y = self.output(y)
        return y
