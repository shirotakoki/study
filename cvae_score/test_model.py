from __future__ import print_function
import argparse
import torch
import torch.utils.data
import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(1)
device = torch.device("cpu")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(5, 3)
        self.fc1 = nn.Linear(3, 5)

    def forward(self, x):
        h1 = F.sigmoid(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        return h2
