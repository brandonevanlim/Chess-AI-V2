import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """One residual block: two convolutions with a skip connection.
    
    The skip connection (x + residual) allows gradients to flow
    directly through without vanishing — this is why deep networks
    train much better with residual connections than without.
    """
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
        return self.relu(x + residual)  # skip connection

class ChessCNN(nn.Module):
    """CNN that predicts the next move from a board position.
    
    Architecture:
      Input conv: 12 channels → 64 channels, preserves 8x8 spatial dims
      Residual blocks: learn spatial patterns (piece coordination, etc.)
      Policy head: flatten → two linear layers → 4096 logits
    
    Output: raw logits over 4096 possible (from_sq, to_sq) pairs.
    No softmax here — CrossEntropyLoss expects raw logits.
    """
    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
        # After res blocks: (batch, 64, 8, 8) → flatten → 4096
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 4096)  # 64 * 64 possible from/to square pairs
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        return self.policy_head(x)  # raw logits, no softmax
