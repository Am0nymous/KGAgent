# encoding: utf-8
import torch
import torch.nn.functional as F
from .utils import euclidean_dist, normalize, cosine_dist, cosine_sim


def domain_SCT_loss(embedding, domain_labels, domain_mask, norm_feat, type):
    if norm_feat: embedding = normalize(embedding, axis=-1)

    unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        if i!=0:
            embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = torch.mean(feat, 0)
        if type == 'euclidean':
            loss = torch.mean(euclidean_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine':
            loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine_sim':
            feat = torch.stack([i for i in feat if i.sum() != 0], dim=0)
            loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)
    try:
        loss_all = torch.mean(torch.stack(loss_all))
    except:
        print("SCT LOSS meets a special sample")
    return loss_all
