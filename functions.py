import time
import random
import numpy as np
import logging
import torch


def common_process_opt(opt):
    if opt.seed > 0:
        random_seed = random.randint(88,500)
        opt.seed = random_seed
        with open('./seed.txt', 'w') as fw:
            fw.write(str(opt.seed))
        set_seed(opt.seed)

    return opt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def time_since(start_time):
    return time.time() - start_time

def read_tokenized_src_file(path, remove_title_eos=True):
    tokenized_train_src = []
    for line_idx, src_line in enumerate(open(path, 'r')):
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_title_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        tokenized_train_src.append(src_word_list)
    return tokenized_train_src


def read_tokenized_trg_file(path):
    data = []
    with open(path) as f:
        for line in f:
            trg_list = line.strip().split(';')
            trg_word_list = [trg.split(' ') for trg in trg_list]
            data.append(trg_word_list)
    return data

def read_src_and_trg_files(src_file, trg_file, is_train, remove_title_eos=True):
    tokenized_train_src = []
    tokenized_train_trg = []
    filtered_cnt = 0
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        if (len(src_line.strip()) == 0) and is_train:
            continue
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_title_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        trg_list = trg_line.strip().split(';')
        trg_word_list = [trg.split(' ') for trg in trg_list]
        if is_train:
            if len(src_word_list) > 400 or len(trg_word_list) > 14:
                filtered_cnt += 1
                continue
        tokenized_train_src.append(src_word_list)
        tokenized_train_trg.append(trg_word_list)

    assert len(tokenized_train_src) == len(
        tokenized_train_trg), 'the number of records in source and target are not the same'

    logging.info("%d rows filtered" % filtered_cnt)

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    return tokenized_train_pairs

