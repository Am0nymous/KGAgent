import torch
import torch.nn as nn
from torch.distributions import Categorical
import functools
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_process(vocab_logits, copy_logits,
                   tgt_s_num, tgt_s_len, b_size,
                   nograd, action, deterministic):
    action_logit = (vocab_logits, copy_logits)
    if nograd == True:
        action = action_from_logits(action_logit, deterministic=False)
    else:
        action = action
    action_logprob, action_entropy = action_stats(action_logit, action)
    action_logit = \
        (action_logit[0].view(b_size, tgt_s_num, tgt_s_len, action_logit[0].size(-1)),
         action_logit[1].view(b_size, tgt_s_num, tgt_s_len, action_logit[1].size(-1)))
    action_logprob = \
        action_logprob.view(b_size, tgt_s_num, tgt_s_len, action_logprob.size(-1))
    return action, action_logit, action_logprob, action_entropy

def action_from_logits(logits, deterministic=True):
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))
    return torch.stack(actions).transpose(1, 0)

def action_stats(logits, action):
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))
    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)

def _get_distributions(action_logits_t, action_logits_r):
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)
    return distribution_t, distribution_r

def _get_actions(distribution_t, distribution_r, deterministic=True):
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r

def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)
    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()
    return [logprob_t, logprob_r], [entropy_t, entropy_r]

def cat(list_of_tensors, dim=0):
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)



def step(finished_word_state, input_feed_w):
    finished_word_state_list = \
        [finished_word_state[tmp_idx] for tmp_idx in range(len(finished_word_state))]
    saved_input_feed_w = torch.stack(finished_word_state_list, dim=0)

    input_feed_w += [saved_input_feed_w]
    return input_feed_w

def relative_reward(reward_1, reward_2):
    better = (reward_1 > reward_2).float() * 0.5
    same = (reward_1 == reward_2).float() * 0.1
    worse = (reward_1 < reward_2).float() * 0.6

    reward = better - worse - same
    return reward

def reward_distance(outputs, gt_embeddings, prev_dist=None):
    dist = torch.cosine_similarity(outputs.view(gt_embeddings.size(0), gt_embeddings.size(1), -1, gt_embeddings.size(-1)),
                                gt_embeddings, dim=-1).permute(2,0,1)
    if prev_dist is not None:
        better = (dist < prev_dist).float() * 0.5
        same = (dist < prev_dist).float() * 0.1
        worse = (dist < prev_dist).float() * 0.6

        reward = better - worse - same
        return reward, dist
    else:
        return torch.zeros_like(dist), dist
