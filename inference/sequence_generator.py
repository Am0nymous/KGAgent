import torch
import pykp.utils.io as io
from inference.beam import Beam
from inference.beam import GNMTGlobalScorer

EPS = 1e-8


class SequenceGenerator(object):
    def __init__(self,
                 model,
                 eos_idx,
                 bos_idx,
                 pad_idx,
                 beam_size,
                 max_sequence_length,
                 copy_attn=False,
                 include_attn_dist=True,
                 length_penalty_factor=0.0,
                 coverage_penalty_factor=0.0,
                 length_penalty='none',
                 coverage_penalty='none',
                 cuda=True,
                 n_best=None,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[]
                 ):
        self.model = model
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_penalty_factor = length_penalty_factor
        self.coverage_penalty_factor = coverage_penalty_factor
        self.include_attn_dist = include_attn_dist
        self.coverage_penalty = coverage_penalty
        self.copy_attn = copy_attn
        self.global_scorer = GNMTGlobalScorer(length_penalty_factor, coverage_penalty_factor, coverage_penalty,
                                              length_penalty)
        self.cuda = cuda
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        if n_best is None:
            self.n_best = self.beam_size
        else:
            self.n_best = n_best

    @classmethod
    def from_opt(cls, model, opt):
        return cls(model,
                   bos_idx=opt.vocab["word2idx"][io.BOS_WORD],
                   eos_idx=opt.vocab["word2idx"][io.EOS_WORD],
                   pad_idx=opt.vocab["word2idx"][io.PAD_WORD],
                   beam_size=opt.beam_size,
                   max_sequence_length=opt.max_length,
                   copy_attn=opt.copy_attention,
                   length_penalty_factor=opt.length_penalty_factor,
                   coverage_penalty_factor=opt.coverage_penalty_factor,
                   length_penalty=opt.length_penalty,
                   coverage_penalty=opt.coverage_penalty,
                   cuda=opt.gpuid > -1,
                   n_best=opt.n_best,
                   block_ngram_repeat=opt.block_ngram_repeat,
                   ignore_when_blocking=opt.ignore_when_blocking)

    def beam_search(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx):
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        memory_bank = self.model.encoder(src, src_lens, src_mask)
        max_num_oov = max([len(oov) for oov in oov_lists])
        memory_bank = memory_bank.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1)
        src_oov = src_oov.repeat(self.beam_size, 1)
        state = self.model.decoder.init_state(memory_bank, src_mask)
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])
        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer,
                          pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx,
                          block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens)
                     for _ in range(batch_size)]

        def var(a):
            return torch.tensor(a, requires_grad=False)

        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                                .t().contiguous().view(-1, 1))
            if self.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.decoder.vocab_size - 1), word2idx[io.UNK_WORD])
            if t > 1:
                decoder_inputs = torch.cat([decoder_inputs, decoder_input], -1)
            else:
                decoder_inputs = decoder_input

            decoder_dist, attn_dist = self.model.decoder(decoder_inputs, state, src_oov, max_num_oov)
            decoder_dist = decoder_dist.squeeze(1)
            attn_dist = decoder_dist.squeeze(1)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)
            attn_dist = attn_dist.view(beam_size, batch_size, -1)

            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])

        result_dict = self._from_beam(beam_list)
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(
                hyps)
            ret['scores'].append(scores)
            ret["attention"].append(attn)
        return ret
