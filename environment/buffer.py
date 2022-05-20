import torch
import numpy as np
import functools

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA =0.5
GAE_LAMBDA = 0.95
def cat(list_of_tensors, dim=0):
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)


def catcat(list_of_lists_of_tensors, dim_outer=0, dim_inner=0):
    return cat([cat(inner_list, dim_inner) for inner_list in list_of_lists_of_tensors], dim_outer)


def discounted(vals, gamma=0.99):
    G = 0
    discounted = torch.zeros_like(vals)
    for i in np.arange(vals.shape[-1]-1, -1, -1):
        G = vals[..., i] + gamma * G
        discounted[..., i] = G
    return discounted


def advantage(rewards, values):
    if GAE_LAMBDA == 0:
        returns = discounted(rewards, GAMMA)
        advantage = returns - values
    else:
        values = torch.cat([values, torch.zeros((values.shape[0], 1, 1)).cuda()], dim=2)
        deltas = rewards + GAMMA * values[..., 1:] - values[..., :-1]
        advantage = discounted(deltas, GAMMA * GAE_LAMBDA)
    return advantage


class Buffer:
    def __init__(self):
        self.count = 0
        self.trg_num = 0
        self.first_final_reward = 0
        self.first_rewards_vocab = 0
        self.first_rewards_src = 0
        self.first_value = 0
        self.rewards = []
        self.rewards_final = []
        self.rewards_vocab = []
        self.rewards_src = []
        self.values = []
        self.actions = []
        self.action_logprobs = []
        self.action_vocab_logits = []
        self.action_src_logits = []

    def __len__(self):
        return self.count

    def start_trajectory(self):
        self.count += 1

    def log_step(self, values, reward_final, reward_vocab, reward_src):
        self.rewards_final.append(reward_final)
        self.rewards_vocab.append(reward_vocab)
        self.rewards_src.append(reward_src)
        self.values.append(values)
        if len(self.rewards_final) >= 2:
            self.rewards_final[0] = self.first_final_reward * 0.999 + reward_final * 0.001
            self.rewards_final.pop()
            self.rewards_vocab[0] = self.first_rewards_vocab * 0.999 + reward_vocab * 0.001
            self.rewards_vocab.pop()
            self.rewards_src[0] = self.first_rewards_src * 0.999 + reward_vocab * 0.001
            self.rewards_src.pop()
            self.values[0] = self.first_value * 0.999 + values * 0.001
            self.values.pop()


    def get_returns_and_advantages(self, rewards, values):
        returns = [discounted(rewards[:, i, :].unsqueeze(1), GAMMA).transpose(2, 1)
                   for i in range(rewards.size(1))]
        advantages = [advantage(rewards[:, i, :].unsqueeze(1), values[:, i, :].unsqueeze(1)).transpose(2, 1)
                      for i in range(rewards.size(1))]
        returns = cat([return_single.unsqueeze(0)for return_single in returns], dim=0).permute(3, 1, 0, 2)
        advantages = cat([return_single.unsqueeze(0) for return_single in advantages], dim=0).permute(3, 1, 0, 2)

        return returns, advantages

    def get_samples(self):
        samples = [self.rewards_final,
                   self.rewards_vocab,
                   self.rewards_src,
                   self.values,
                   self.actions,
                   self.action_vocab_logits,
                   self.action_src_logits,
                   self.action_logprobs]
        samples += self.get_returns_and_advantages(self.rewards_final[0], self.values[0])
        samples += self.get_returns_and_advantages(self.rewards_vocab[0], self.values[0])
        samples += self.get_returns_and_advantages(self.rewards_src[0], self.values[0])
        return [cat(sample) for sample in samples], self.trg_num

    def returns_and_advantages(self):
        rewards_final = self.get_returns_and_advantages(self.rewards_final[0], self.values[0])
        rewards_vocab = self.get_returns_and_advantages(self.rewards_vocab[0], self.values[0])
        rewards_src = self.get_returns_and_advantages(self.rewards_src[0], self.values[0])
        return cat(rewards_final), cat(rewards_vocab), cat(rewards_src),

    def clear(self):
        self.count = 0
        self.trg_num = 0
        self.rewards_final.clear()
        self.rewards_vocab.clear()
        self.rewards_src.clear()
        self.values.clear()

