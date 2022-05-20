import random
import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pykp.utils import torchutils


def _get_Z(self, domain, vec, t):
    bank = self.get_attr(domain, "memory_bank")
    Z = torchutils.contrastive_sim_z(vec, bank, tao=t)
    return Z

def _get_all_dot_products(self, domain, vec):
    assert len(vec.size()) == 2
    bank = self.get_attr(domain, "memory_bank")
    return torch.matmul(vec, torch.transpose(bank, 1, 0))



def compute_Indmain_loss(outputs, total_tokens, t=0.05):
    loss = torch.Tensor([0]).cuda()
    batch_size = outputs.size(0)

    for i in range(batch_size):
        batch_labels = torch.tensor(list(range(6)))
        output = outputs[i]
        dot_exp = torch.exp(
            torch.sum(output * output, dim=-1)
        )

        z = torchutils.contrastive_sim_z(output, output, tao=0.1)
        p = dot_exp / z
        loss = loss - torch.sum(torch.log(p))
    loss /= total_tokens
    return loss

def compute_Outdomain_loss(src_embeddings, trg_embeddings, total_tokens, t=0.05):
    loss = torch.Tensor([0]).cuda()
    batch_size = trg_embeddings.size(0)
    for i in range(batch_size):
        src_embedding = src_embeddings[i]
        trg_embedding = trg_embeddings[i]
        p = torchutils.contrastive_sim_z(src_embedding, trg_embedding, tao=0.1)
        z = torch.sum(p, dim=-1, keepdim=True)
        p = p / z.unsqueeze(1)
        loss = loss - torch.sum(p * torch.log(p))
    loss /= total_tokens

    return loss




