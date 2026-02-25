# -*- coding: utf-8 -*-

# Brief: Corpus for model

import glob
import os
import random
import sys
from codecs import open
from collections import Counter

import numpy as np

# 关键词纠错数据：每行 错误文本<mask_99>关键词1, 关键词2, ...<mask_99>正确文本
KEYWORD_SEP = "<mask_99>"
KEYWORD_USER_TEMPLATE = "错误文本：{error}\n候选关键词：{keywords}"

# Define constants associated with the usual special tokens.
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                print('error', line)
    return dict_data


def read_vocab(input_texts, max_size=None, min_count=0):
    token_counts = Counter()
    special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for texts in input_texts:
        for token in texts:
            token_counts.update(token)
    # Sort word count by value
    count_pairs = token_counts.most_common()
    vocab = [k for k, v in count_pairs if v >= min_count]
    # Insert the special tokens to the beginning
    vocab[0:0] = special_tokens
    full_token_id = list(zip(vocab, range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    return vocab2id


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_bert_data(path, use_segment, num_examples=None):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    src_trg_lines = []

    for line in lines[:num_examples]:
        terms = line.split('\t')
        if len(terms) != 2:
            continue
        src = terms[0].replace(' ', '') if use_segment else terms[0]
        trg = terms[1].replace(' ', '') if use_segment else terms[1]
        src_trg_lines.append([src, trg])
    return src_trg_lines


def _read_keyword_lines_from_dir(dir_path):
    """从目录下所有 txt 中读取并解析 错误<mask_99>关键词<mask_99>正确 行，返回 [(error, keywords, target), ...]。"""
    if not os.path.isdir(dir_path):
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(KEYWORD_SEP)
                if len(parts) != 3:
                    continue
                error_text = parts[0].strip()
                keywords_str = parts[1].strip()
                target_text = parts[2].strip()
                out.append((error_text, keywords_str, target_text))
    return out


def load_keyword_correction_data(pos_dir, neg_dir, neg_ratio=1.0, seed=42):
    """
    加载「上下文关键词」纠错数据：正例全用，负例按 neg_ratio 倍正例数量随机采样后与正例打乱。
    返回 list of [input_text, target_text]，其中 input_text 已含「错误文本」与「候选关键词」模板。
    """
    random.seed(seed)
    pos_list = _read_keyword_lines_from_dir(pos_dir)
    neg_list = _read_keyword_lines_from_dir(neg_dir)
    n_pos = len(pos_list)
    n_neg_sample = max(0, int(round(n_pos * neg_ratio)))
    if n_neg_sample > 0 and neg_list:
        neg_sampled = random.sample(neg_list, min(n_neg_sample, len(neg_list)))
    else:
        neg_sampled = []
    # 构建 (input_text, target_text)：input_text 为模型看到的用户输入（错误+关键词）
    rows = []
    for error, keywords, target in pos_list + neg_sampled:
        input_text = KEYWORD_USER_TEMPLATE.format(error=error, keywords=keywords)
        rows.append([input_text, target])
    random.shuffle(rows)
    return rows


def build_keyword_input(error_text, keywords_list):
    """推理时构造一条关键词纠错输入（与训练时一致）。keywords_list 为 list 或已用逗号拼接的 str。"""
    if isinstance(keywords_list, (list, tuple)):
        keywords_str = ", ".join(str(k) for k in keywords_list)
    else:
        keywords_str = str(keywords_list).strip()
    return KEYWORD_USER_TEMPLATE.format(error=error_text.strip(), keywords=keywords_str)


def create_dataset(path, num_examples=None):
    """
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    :param path:
    :param num_examples:
    :return:
    """
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(s) for s in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def preprocess_sentence(sentence):
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    return [SOS_TOKEN] + sentence.lower().split() + [EOS_TOKEN]


def show_progress(curr, total, time=""):
    prog_ = int(round(100.0 * float(curr) / float(total)))
    dstr = '[' + '>' * int(round(prog_ / 4)) + ' ' * (25 - int(round(prog_ / 4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time + '\r')
    sys.stdout.flush()


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)  # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def prepare_data(seqs, max_length=None):
    if max_length:
        seqs = [seq[:max_length] for seq in seqs]
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(src_sentences, trg_sentences, batch_size, max_length=None):
    minibatches = get_minibatches(len(src_sentences), batch_size)
    examples = []
    for minibatch in minibatches:
        mb_src_sentences = [src_sentences[t] for t in minibatch]
        mb_trg_sentences = [trg_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_src_sentences, max_length)
        mb_y, mb_y_len = prepare_data(mb_trg_sentences, max_length)
        examples.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return examples


def one_hot(src_sentences, trg_sentences, src_dict, trg_dict, sort_by_len=True):
    """vector the sequences.
    """
    out_src_sentences = [[src_dict.get(w, 0) for w in sent] for sent in src_sentences]
    out_trg_sentences = [[trg_dict.get(w, 0) for w in sent] for sent in trg_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # sort length
    if sort_by_len:
        sorted_index = len_argsort(out_src_sentences)
        out_src_sentences = [out_src_sentences[i] for i in sorted_index]
        out_trg_sentences = [out_trg_sentences[i] for i in sorted_index]

    return out_src_sentences, out_trg_sentences
