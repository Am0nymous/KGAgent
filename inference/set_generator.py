import pykp.utils.io as io
import torch

EPS = 1e-8

class SetGenerator(object):
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_opt(cls, model, opt):
        return cls(model)

    def inference(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx):
        self.model.eval()
        batch_size = src.size(0)
        max_kp_num = self.model.decoder.max_kp_num
        max_kp_len = self.model.decoder.max_kp_len
        vocab_size = self.model.decoder.vocab_size
        memory_bank = self.model.encoder(src, src_lens, src_mask)
        state = self.model.decoder.init_state(memory_bank, src_mask)
        control_embed = self.model.decoder.forward_seg(state)
        max_num_oov = max([len(oov) for oov in oov_lists])
        attn_dict_list = []
        decoder_score_list = []
        output_tokens = src.new_zeros(batch_size, max_kp_num, max_kp_len + 1)
        output_tokens[:, :, 0] = word2idx[io.BOS_WORD]
        for t in range(1, max_kp_len+1):
            decoder_inputs = output_tokens[:, :, :t]
            decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(vocab_size - 1), word2idx[io.UNK_WORD])
            decoder_dist, attn_dist, _, _, _ = self.model.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
            attn_dict_list.append(attn_dist.reshape(batch_size, max_kp_num, 1, -1))
            decoder_score_list.append(decoder_dist.max(-1)[0].reshape(batch_size, max_kp_num, 1))
            _, tokens = decoder_dist.max(-1)
            output_tokens[:, :, t] = tokens

        output_tokens = output_tokens[:, :, 1:].reshape(batch_size, max_kp_num*max_kp_len)[:, None]
        attn_dicts = torch.cat(attn_dict_list, -2).reshape(batch_size, max_kp_num*max_kp_len, -1)[:, None]
        decoder_scores = torch.cat(decoder_score_list, -1).reshape(batch_size, max_kp_num * max_kp_len)[:, None]
        result_dict = {"predictions": [], "attention": [], "decoder_scores": []}
        for b in range(batch_size):
            result_dict["predictions"].append(output_tokens[b])
            result_dict["attention"].append(attn_dicts[b])
            result_dict["decoder_scores"].append(decoder_scores[b])
        return result_dict
1269
826
1876
