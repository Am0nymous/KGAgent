import logging
import math
import os
import sys
import time

import pykp.utils.io as io
from inference.evaluate import evaluate_loss
from pykp.utils.label_assign import hungarian_assign
from pykp.utils.masked_loss import masked_cross_entropy
from pykp.utils.domain_SCT_loss import domain_SCT_loss
from pykp.utils.triplet_loss import triplet_loss
from pykp.utils.domian_loss import compute_Outdomain_loss, compute_Indmain_loss
from utils.functions import time_since
from utils.report import export_train_and_valid_loss
from utils.statistics import LossStatistics
from environment.buffer import Buffer
from environment.environment import *
from pykp.utils.utils import normalize

EPS = 1e-8

def train_model(model, optimizer,
                critic_model, coptimizer,
                train_data_loader, valid_data_loader, opt):
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0

    model.train()
    buffer = Buffer()
    buffer.start_trajectory()

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if early_stop_flag:
            break
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            batch_loss_stat = \
                train_one_batch(buffer, batch, model, optimizer, critic_model, coptimizer, opt)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)


            if total_batch % opt.report_every == 0:
                current_train_ppl = report_train_loss_statistics.ppl()
                current_train_loss = report_train_loss_statistics.xent()
                current_train_total_loss, \
                current_train_ppo_loss,   \
                current_train_pre_loss,   \
                current_train_ab_loss, \
                current_train_SCT_loss, \
                current_train_da_loss= report_train_loss_statistics.logging_loss()
                logging.info(
                    "\n Epoch %d; batch: %d; total batch: %d，avg training ppl: %.3f, loss: %.3f, "
                    "\n total_loss: %.3f, ppo_loss: %.3f, pre_loss: %.3f, ab_loss: %.3f, "
                    "\n loss_SCT: %.3f, loss_da: %.8f \n"
                    % (epoch, batch_i, total_batch, current_train_ppl, current_train_loss,
                       current_train_total_loss,
                       current_train_ppo_loss,
                       current_train_pre_loss,
                       current_train_ab_loss,
                       current_train_SCT_loss,
                       current_train_da_loss))
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and
                         total_batch % opt.checkpoint_interval == 0):
                    valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    model.train()

                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    logging.info("Enter check point!")
                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()

                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (
                                epoch, batch_i, total_batch))
                        exit()
                    if current_valid_loss < best_valid_loss:
                        logging.info("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0
                        saved_model = {
                            "model":model.state_dict(),
                            "critic_model":critic_model.state_dict()
                        }
                        check_pt_model_path = os.path.join(opt.model_path, 'best_model.pt')
                        torch.save(
                            saved_model,
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        num_stop_dropping += 1
                        logging.info("Valid loss does not drop, patience: %d/%d" % (
                            num_stop_dropping, opt.early_stop_tolerance))

                        for i, param_group in enumerate(optimizer.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        ' * avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        ' * avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info(
                            'Have not increased for %d check points, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break
                    report_train_loss_statistics.clear()

    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl,
                                opt.checkpoint_interval, train_valid_curve_path)


def train_one_batch(buffer, batch, model, optimizer, critic_model, coptimizer, opt):

    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
    trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch

    max_num_oov = max([len(oov) for oov in oov_lists])
    batch_size = src.size(0)
    _, tgt_s_num, tgt_s_len = trg_oov.size()

    word2idx = opt.vocab['word2idx']
    target = trg_oov if opt.copy_attention else trg

    optimizer.zero_grad()
    coptimizer.zero_grad()
    start_time = time.time()
    if opt.fix_kp_num_len:
        y_t_init = target.new_ones(batch_size, opt.max_kp_num, 1) * word2idx[io.BOS_WORD]
        if opt.set_loss:
            model.eval()
            critic_model.eval()
            with torch.no_grad():
                nograd = True
                action = None
                deterministic = False

                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                control_embed = model.decoder.forward_seg(state)

                input_tokens = src.new_zeros(batch_size, opt.max_kp_num, target.size(-1) + 1)
                decoder_dists = []
                vocab_dists = []
                src_dists = []
                trg_logits = []
                input_tokens[:, :, 0] = word2idx[io.BOS_WORD]
                for t in range(1, target.size(-1) + 1):
                    decoder_inputs = input_tokens[:, :, :t]
                    decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(opt.vocab_size - 1),
                                                                word2idx[io.UNK_WORD])
                    decoder_dist, _, vocab_dist_softmax, attn_dist_softmax, trg_logit = model.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
                    input_tokens[:, :, t] = decoder_dist.argmax(-1)
                    decoder_dists.append(decoder_dist.reshape(batch_size, opt.max_kp_num, 1, -1))
                    vocab_dists.append(vocab_dist_softmax.reshape(batch_size, opt.max_kp_num, 1, -1))
                    src_dists.append(attn_dist_softmax.reshape(batch_size, opt.max_kp_num, 1, -1))
                    trg_logits.append(trg_logit.reshape(batch_size, opt.max_kp_num, 1, -1))

                decoder_dists_pre_ab = torch.cat(decoder_dists, -2)[:, :, :opt.assign_steps, :]

                trg_logits = torch.cat(trg_logits, -2)
                decoder_dists_old = torch.cat(decoder_dists, -2)
                vocab_logits = torch.cat(vocab_dists, -2).view(-1, vocab_dist_softmax.size(-1))
                src_logits = torch.cat(src_dists, -2).view(-1, attn_dist_softmax.size(-1))

                action, action_logit, action_logprob, action_entropy = \
                    action_process(vocab_logits, src_logits,
                                   tgt_s_num, tgt_s_len, batch_size,
                                   nograd, action, deterministic)

                values = critic_model(trg_logits)
                values = values.squeeze()

                if opt.seperate_pre_ab:
                    mid_idx = opt.max_kp_num // 2
                    pre_reorder_index = hungarian_assign(decoder_dists_pre_ab[:, :mid_idx],
                                                         target[:, :mid_idx, :opt.assign_steps],
                                                         ignore_indices=[word2idx[io.NULL_WORD],
                                                                         word2idx[io.PAD_WORD]])
                    target[:, :mid_idx] = target[:, :mid_idx][pre_reorder_index]
                    trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                    ab_reorder_index = hungarian_assign(decoder_dists_pre_ab[:, mid_idx:],
                                                        target[:, mid_idx:, :opt.assign_steps],
                                                        ignore_indices=[word2idx[io.NULL_WORD],
                                                                        word2idx[io.PAD_WORD]])
                    target[:, mid_idx:] = target[:, mid_idx:][ab_reorder_index]
                    trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]
                else:
                    reorder_index = hungarian_assign(decoder_dists_pre_ab, target[:, :, :opt.assign_steps],
                                                     [word2idx[io.NULL_WORD],
                                                      word2idx[io.PAD_WORD]])
                    target = target[reorder_index]
                    trg_mask = trg_mask[reorder_index]


            model.train()
            critic_model.train()

        input_tgt = torch.cat([y_t_init, target[:, :, :-1]], dim=-1)
        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        control_embed = model.decoder.forward_seg(state)

        input_tgt = input_tgt.masked_fill(input_tgt.gt(opt.vocab_size - 1), word2idx[io.UNK_WORD])
        decoder_dist_new, attn_dist_new, vocab_logits_new, src_logits_new, trg_logit_new = \
            model.decoder(input_tgt, state, src_oov, max_num_oov, control_embed)

        nograd = False
        deterministic = True
        vocab_logits_new = \
            vocab_logits_new.view(batch_size, tgt_s_num, tgt_s_len, vocab_logits_new.size(-1))\
                .view(-1, vocab_logits_new.size(-1))
        src_logits_new = \
            src_logits_new.view(batch_size, tgt_s_num, tgt_s_len, src_logits_new.size(-1))\
                          .view(-1, src_logits_new.size(-1))

        action_new, action_logit_new, action_logprobs_new, action_entropy_new = \
            action_process(vocab_logits_new,
                           src_logits_new,
                           tgt_s_num, tgt_s_len, batch_size,
                           nograd, action, deterministic)

        values_new = critic_model(trg_logit_new)
        values_new = values_new.squeeze()

        _, loss_nll_new = masked_cross_entropy(decoder_dist_new, target, trg_mask)
        _, loss_nll_old = \
            masked_cross_entropy(decoder_dists_old.view(batch_size, -1, decoder_dists_old.size(-1)),
                                 target,
                                 trg_mask)

        _, loss_vocab_new = \
            masked_cross_entropy(vocab_logits_new.view(batch_size, -1, vocab_logits_new.size(-1)),
                                                 target,
                                                 trg_mask)
        _, loss_vocab_old = \
            masked_cross_entropy(vocab_logits.view(batch_size, -1, vocab_logits.size(-1)),
                                 target,
                                 trg_mask)

        _, loss_src_new = masked_cross_entropy(src_logits_new.view(batch_size, -1, src_logits_new.size(-1)),
                                               target, trg_mask)
        _, loss_src_old = \
            masked_cross_entropy(src_logits.view(batch_size, -1, src_logits.size(-1)),
                                 target,
                                 trg_mask)

        relative_reward_final = relative_reward(loss_nll_old, loss_nll_new)
        relative_reward_vocab = relative_reward(loss_src_old, loss_src_new)
        relative_reward_src = relative_reward(loss_vocab_old, loss_vocab_new)

        buffer.log_step(values,
                        relative_reward_final,
                        relative_reward_vocab,
                        relative_reward_src
                        )

        (returns, advantages), (returns_vocab, advantages_vocab), (returns_src, advantages_src) = \
            buffer.returns_and_advantages()
    else:
        y_t_init = trg.new_ones(batch_size, 1) * word2idx[io.BOS_WORD]
        input_tgt = torch.cat([y_t_init, trg[:, :-1]], dim=-1)
        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        decoder_dist_new, attention_dist, vocab_dist = model.decoder(input_tgt, state, src_oov, max_num_oov)

    forward_time = time_since(start_time)
    start_time = time.time()
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    CLIP_VALUE = False
    C_VALUE = 0.3
    C_ENTROPY = 1e-3
    alpha = 2
    total_trg_tokens = trg_mask.sum().item()
    if opt.RL:

        ratio_vocab = torch.exp(action_logprobs_new[:, :, :, 0] -
                                action_logprob[:, :, :, 0]).cuda()
        ratio_src = torch.exp(action_logprobs_new[:, :, :, 1] -
                              action_logprob[:, :, :, 1]).cuda()

        policy_vocab_loss = \
            -torch.min(ratio_vocab * advantages.float().squeeze() * trg_mask,
                       ratio_vocab.clamp(1 - CLIP_EPS,
                                         1 + CLIP_EPS) * advantages.float().squeeze() * trg_mask).sum() / total_trg_tokens
        policy_src_loss = \
            -torch.min(ratio_src * advantages.float().squeeze() * trg_mask,
                       ratio_src.clamp(1 - CLIP_EPS,
                                       1 + CLIP_EPS) * advantages.float().squeeze() * trg_mask).sum() / total_trg_tokens

        value_loss_ppo = \
            (values_new.unsqueeze(-1).contiguous().view(-1, 1) -
             returns.unsqueeze(-1).view(-1, 1).float()).pow(2) * (
                trg_mask.unsqueeze(-1).reshape(-1, 1))
        value_loss = value_loss_ppo.sum() / total_trg_tokens

        if CLIP_VALUE:
            values_clipped_ppo = values + (values_new - values) \
                .clamp(-CLIP_EPS, CLIP_EPS)
            losses_v_clipped_ppo = (values_clipped_ppo.view(-1, 1) - returns).pow(2)
            value_loss_ppo = torch.max(value_loss_ppo, losses_v_clipped_ppo)

        entropy_loss = action_entropy_new * (trg_mask.unsqueeze(-1).reshape(-1, 1).repeat(1, 2))
        entropy_loss = entropy_loss.sum() / total_trg_tokens

        ppo_loss = \
            policy_vocab_loss + policy_src_loss + value_loss * C_VALUE - entropy_loss * C_ENTROPY
        loss_ppo = ppo_loss * alpha


    trg_oov_mask = trg_oov.ne(word2idx['<pad>']).ne(0).float().cuda()
    domain_labels = []
    domain_labels_triple = []
    for i in range(trg_oov_mask.size(0)):
        k = 0
        for j in range(trg_oov_mask.size(1)):
            if trg_oov_mask[i][j].sum()==1:
                domain_labels.append([0]*(trg_oov.size(-1)))
                domain_labels_triple.append([0] * (trg_oov.size(-1)))
            else:
                k = k + 1
                len_valid = int((trg_oov_mask[i][j].sum()-1).item())
                domain_labels.append([k]*len_valid+[0]*(trg_oov.size(-1)-len_valid))
                domain_labels_triple.append([0] * (trg_oov.size(-1)))

    domain_labels = [b for a in domain_labels for b in a]
    domain_labels = torch.tensor(domain_labels).view(batch_size, -1)

    domain_labels_triple = [b for a in domain_labels_triple for b in a]
    domain_labels_triple = torch.tensor(domain_labels_triple).view(batch_size, -1)

    domain_mask = domain_labels.ne(0).int().cuda()
    total_src_tokens = src_mask.sum().item()
    opt.loss_outdomain = True
    if opt.loss_outdomain:
        memory_bank = memory_bank * (src_mask.unsqueeze(-1).repeat(1,1,memory_bank.size(-1)))
        trg_logit_new = trg_logit_new * (domain_mask.unsqueeze(-1).repeat(1,1,trg_logit_new.size(-1)))

        src_embeddings = normalize(memory_bank, axis=-1)
        trg_embeddings = normalize(trg_logit_new, axis=-1)

        lo_outdomain = compute_Outdomain_loss(src_embeddings, trg_embeddings, total_src_tokens)

    opt.loss_indomian =  False
    if opt.loss_indomian:
        memory_bank = memory_bank * (src_mask.unsqueeze(-1).repeat(1, 1, memory_bank.size(-1)))
        trg_logit_new = trg_logit_new * (domain_mask.unsqueeze(-1).repeat(1, 1, trg_logit_new.size(-1)))

        src_embeddings = normalize(memory_bank, axis=-1)
        trg_embeddings = normalize(trg_logit_new, axis=-1)

        loss_indomain = compute_Indmain_loss(trg_embeddings, total_trg_tokens)

    opt.SCT = False
    if opt.SCT:
        loss_SCTs = []
        for i in range(trg_logit_new.size(0)):
            trg_logits_i = trg_logit_new[i].view(-1, trg_logit_new.size(-1))
            domain_label_i = domain_labels[i].view(-1, 1).squeeze(-1)
            domain_mask_i = domain_mask[i].view(-1, 1).squeeze(-1)
            domain_sct = domain_SCT_loss(trg_logits_i,
                                         domain_label_i,
                                         domain_mask_i,
                                         True,
                                         'cosine_sim')
            if domain_sct == []:
                continue
            else:
                loss_SCTs.append(domain_sct)
        loss_SCT = torch.stack(loss_SCTs).mean()

    opt.tripleloss = False
    if opt.tripleloss:
        loss_triples = []
        for i in range(trg_logit_new.size(0)):
            trg_logits_i = trg_logit_new[i].view(-1, trg_logit_new.size(-1))
            domain_label_i = domain_labels_triple[i].view(-1, 1).squeeze(-1)

            trg_oov_i = trg_oov[i].reshape((trg_oov[i].size(0)) * (trg_oov[i].size(-1)))
            loss_triplet_ = triplet_loss(trg_logits_i,
                                         domain_label_i,
                                         trg_oov_i,
                                         margin=0.0,
                                         norm_feat=False,
                                         hard_mining=True,
                                         dist_type='euclidean',
                                         loss_type='logistic'
                                         )
            if loss_triplet_ == []:
                continue
            else:
                loss_triples.append(loss_triplet_)

        loss_triplet = torch.stack(loss_triples).mean()

    if opt.fix_kp_num_len:
        if opt.seperate_pre_ab:
            mid_idx = opt.max_kp_num // 2
            pre_loss, _ = masked_cross_entropy(
                decoder_dist_new.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, :mid_idx]\
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, :mid_idx].reshape(batch_size, -1),
                trg_mask[:, :mid_idx].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_pre],
                scale_indices=[word2idx[io.NULL_WORD]])
            ab_loss, _ = masked_cross_entropy(
                decoder_dist_new.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, mid_idx:]
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, mid_idx:].reshape(batch_size, -1),
                trg_mask[:, mid_idx:].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_ab],
                scale_indices=[word2idx[io.NULL_WORD]])
            loss = pre_loss + ab_loss
        else:
            loss, _ = masked_cross_entropy(decoder_dist_new, target.reshape(batch_size, -1),
                                        trg_mask.reshape(batch_size, -1),
                                        loss_scales=[opt.loss_scale],
                                        scale_indices=[word2idx[io.NULL_WORD]])
    else:
        loss = masked_cross_entropy(decoder_dist_new, target, trg_mask)


    loss_compute_time = time_since(start_time)

    total_trg_tokens = trg_mask.sum().item()
    total_trg_sents = src.size(0)
    if opt.loss_normalization == "tokens":
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':
        normalization = total_trg_sents
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    total_loss = loss.div(normalization) + loss_ppo + lo_outdomain

    total_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()
    coptimizer.step()
    loss_triplet = torch.tensor(0)
    loss_SCT = torch.tensor(0)
    stat = LossStatistics(total_loss.item(),
                          lo_outdomain.item(), loss_SCT.item(), ppo_loss.item(),
                          ab_loss.item(), pre_loss.item(),
                          loss.item(), total_trg_tokens,
                          n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)
    return stat
