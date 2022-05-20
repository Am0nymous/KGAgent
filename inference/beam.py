import torch

from inference.penalties import PenaltyBuilder

class Beam:
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):
        self.size = size
        self.tt = torch.cuda if cuda else torch

        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.prev_ks = []

        self.next_ys = [self.tt.LongTensor(size)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        self._eos = eos
        self.eos_top = False

        self.attn = []

        self.finished = []
        self.n_best = n_best

        self.global_scorer = global_scorer
        self.global_state = {}

        self.min_length = min_length

        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

        self.eos_counters = torch.zeros(size, dtype=torch.long).to(self.next_ys[0].device)

    def get_current_tokens(self):

        return self.next_ys[-1]

    def get_current_origin(self):

        return self.prev_ks[-1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def get_hyp(self, timestep, k):
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) -1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])

            k = self.prev_ks[j][k]

        return hyp[::-1], torch.stack(attn)

    def advance(self, word_logits, attn_dist):
        vocab_size = word_logits.size(1)
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_logits)):
                word_logits[k][self._eos] = -1e20
        if len(self.prev_ks) > 0:
            beam_scores = word_logits + self.scores.unsqueeze(1).expand_as(word_logits)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):

                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]

                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20

        else:
            beam_scores = word_logits[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_idx = flat_beam_scores.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores


        prev_k = best_scores_idx // vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_idx - prev_k * vocab_size))
        self.attn.append(attn_dist.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)
        self.update_eos_counter()

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                self.eos_counters[i] += 1
                if self.eos_counters[i] == 1:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))
        if self.next_ys[-1][0] == self._eos and self.eos_counters[0] == 1:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0

            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys)-1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t,k) for _, t, k in self.finished]
        return scores, ks

    def update_eos_counter(self):
        self.eos_counters = self.eos_counters.index_select(0, self.prev_ks[-1])


class GNMTGlobalScorer:
    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(cov_penalty, length_penalty)
        self.cov_penalty = penalty_builder.coverage_penalty()
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty

