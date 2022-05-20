import logging
import os
import random
import shutil
import sys
from subprocess import call
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function



def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def dot(x, y):
    return torch.sum(x * y, dim=-1)


def contrastive_sim(instances, proto=None, tao=0.05):
    if proto is None:
        proto = instances
    ins_ext = instances.unsqueeze(1).repeat(1, proto.size(0), 1)
    sim_matrix = torch.exp(dot(ins_ext, proto) / tao)
    return sim_matrix


def contrastive_sim_z(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return torch.sum(sim_matrix, dim=-1)


def contrastive_prob(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return sim_matrix / torch.sum(sim_matrix, dim=-1).unsqueeze(-1)


def pairwise_distance_2(input_1, input_2):
    assert input_1.size(1) == input_2.size(1)
    dis_vec = input_1.unsqueeze(1) - input_2
    dis = torch.norm(dis_vec, dim=2)
    return dis


def weights_init(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)




def entropy(x, eps=1e-5):
    p = F.softmax(x, dim=-1)
    entropy = -torch.mean(torch.sum(p * torch.log(p + eps), 1))
    return entropy



