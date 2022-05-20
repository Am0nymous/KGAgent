import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

import functools
EPS = 1e-8

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_idx = 0
        self.activation = nn.ReLU()
        self.emb_v = nn.Sequential(
            nn.Linear(512, 512),
            self.activation
        )
        self.value = nn.Linear(512, 1)

    def forward(self, outputs):
        emb_v = self.emb_v(outputs)
        value = self.value(emb_v)
        return value.squeeze()

def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos

def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

