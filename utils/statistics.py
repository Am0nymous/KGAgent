import math

class LossStatistics:
    def __init__(self, total_loss=0.0, loss_da=0.0, loss_SCT=0.0,
                       ppo_loss=0.0, ab_loss=0.0, pre_loss=0.0, loss=0.0,
                       n_tokens=0, n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        self.loss = loss
        self.total_loss = total_loss
        self.loss_da = loss_da
        self.loss_SCT = loss_SCT
        self.ppo_loss = ppo_loss
        self.ab_loss = ab_loss
        self.pre_loss = pre_loss
        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time
        self.interval = 0.0

    def update(self, stat):
        self.loss += stat.loss
        self.total_loss +=stat.total_loss
        self.ppo_loss +=stat.ppo_loss
        self.loss_da += stat.loss_da
        self.loss_SCT += stat.loss_SCT
        self.pre_loss +=stat.pre_loss
        self.ab_loss +=stat.ab_loss
        self.interval +=1
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time

    def xent(self):
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.loss / self.n_tokens

    def ppl(self):
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.loss / self.n_tokens, 100))
    def logging_loss(self):
        return self.total_loss / self.interval, \
               self.ppo_loss / self.interval, \
               self.pre_loss / self.n_tokens, \
               self.ab_loss / self.n_tokens, \
               self.loss_SCT / self.interval, \
               self.loss_da / self.interval

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time
    def clear(self):
        self.loss = 0.0
        self.total_loss = 0.0
        self.ppo_loss = 0.0
        self.loss_da = 0.0
        self.loss_SCT = 0.0
        self.pre_loss = 0.0
        self.ab_loss = 0.0
        self.interval = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0
