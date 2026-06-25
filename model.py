import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class ChessCNN(nn.Module):
    def __init__(self, num_res_blocks=6, channels=96, dropout=0.3): # Changed num_res_blocks from 4 to 6 and channels from 64 to 96 to increase validation accuracy
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 4096)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return self.policy_head(x)
