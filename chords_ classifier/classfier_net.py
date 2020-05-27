import torch.nn as nn
import torch.nn.functional as F

class ClassfierNet(nn.Module):
    def __init__(self, time_dim: int, pitch_dim: int):
        super(ClassfierNet, self).__init__()
        self.fc1 = nn.Linear(time_dim*pitch_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x