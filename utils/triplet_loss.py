# encoding: utf-8

import torch
import torch.nn.functional as F
import copy
from .utils import  euclidean_dist, normalize, cosine_dist

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):

    assert len(dist_mat.size()) == 2
    N = dist_mat.size(0)

    dist_ap = list()
    for i in range(dist_mat.shape[0]):
        dist_ap.append(torch.max(dist_mat[i][is_pos[i]]))
    dist_ap = torch.stack(dist_ap)

    dist_an = list()
    for i in range(dist_mat.shape[0]):
        if is_neg[i].sum()!=0:
            dist_an.append(torch.min(dist_mat[i][is_neg[i]]))
        else:
            dist_an.append(torch.tensor(0).cuda())
    dist_an = torch.stack(dist_an)

    return dist_ap, dist_an

def weighted_example_mining(dist_mat, is_pos, is_neg):
    assert len(dist_mat.size()) == 2

    is_pos = is_pos.float()
    is_neg = is_neg.float()
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, domain_labels, targets, margin, norm_feat, hard_mining, dist_type,
                 loss_type, pos_flag = [0, 0, 1], neg_flag = [0,1,0]):
    if norm_feat: embedding = normalize(embedding, axis=-1)

    trg_zeros = torch.zeros(targets.size(0)).long().cuda()
    targets = torch.where(targets!=6, targets, trg_zeros)
    all_embedding = embedding  # 120, 512    20*6, 512
    all_targets = targets  # 80

    if dist_type == 'euclidean':
        dist_mat = euclidean_dist(all_embedding, all_embedding)  # 120, 120
    elif dist_type == 'cosine':
        dist_mat = cosine_dist(all_embedding, all_embedding)

    N = dist_mat.size(0)

    flag=False
    if flag:
        is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())
        is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())
    else:
        vec1 = copy.deepcopy(all_targets)
        for i in range(N):
            vec1[i] = i
        is_same_keyphrase = vec1.expand(N, N).eq(vec1.expand(N, N).t())
        is_same_instance = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())
        is_same_domain = domain_labels.view(N, 1).expand(N, N).eq(domain_labels.view(N, 1).expand(N, N).t()).cuda()

        set0 = is_same_keyphrase

        set_all = []
        set_all.extend([is_same_instance * (is_same_keyphrase == False)])
        set_all.extend([(is_same_instance == False) * (is_same_domain == True)])
        set_all.extend([is_same_domain == False])

        is_pos = copy.deepcopy(set0)
        is_neg = copy.deepcopy(set0 == False)
        is_neg[:] = False


        for i, bool_flag in enumerate(pos_flag):
            if bool_flag == 1:
                is_pos += set_all[i]

        for i, bool_flag in enumerate(neg_flag):
            if bool_flag == 1:
                is_neg += set_all[i]


    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        if loss_type == 'logistic':
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        elif loss_type == 'hinge':
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    return loss

