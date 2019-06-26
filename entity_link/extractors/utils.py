#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-24 14:25
# @Author  : zhangzhen
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
from typing import List

import numpy as np
import random
import codecs
import pickle
import logging
import copy
import os

logger = logging.getLogger(__name__)


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with codecs.open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    logging.info('vocab_size: {}'.format(len(word2id)))
    return word2id


def random_embedding(vocab_size, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def conlleval(label_predict, label_path, metric_path, eval_perl="./conlleval_rev.pl"):
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                char = b'<spe>' if char == b' ' else char
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def random_split_data(data, seed=1234, mode=0):
    np.random.seed(seed=seed)
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    dev = [data[j] for i, j in enumerate(random_order) if i % 9 == mode]
    train = [data[j] for i, j in enumerate(random_order) if i % 9 != mode]
    return train, dev


def deduplicate_entities(entities: List):
    # (val, offset)
    ent_tree = {}
    offsets = set()
    for (val, offset) in entities:
        length = len(val)
        if offset in ent_tree:
            (_offset, _word, _length) = ent_tree[offset]
            # 新实体大于 库中实体
            if _offset == offset and _offset + _length < offset + length:
                # 替换操作
                for i in range(length):
                    word = val[i]
                    ent_tree[i + offset] = (offset, word, length)
        elif offset + length in ent_tree:
            (_offset, _word, _length) = ent_tree[offset + length]
            if length > _length:
                for i in range(_offset, _offset + _length):
                    del ent_tree[i]

                for i in range(length):
                    word = val[i]
                    ent_tree[i + offset] = (offset, word, length)
        else:
            for i in range(length):
                word = val[i]
                ent_tree[i + offset] = (offset, word, length)

    rs_entities = []
    keySet = set()
    for key, (offset, word, length) in ent_tree.items():
        if offset in keySet:
            pass
        else:
            val = ''.join([ent_tree.get(i)[1] for i in range(offset, offset + length)])
            keySet.add(offset)
            rs_entities.append((val, offset))

    return rs_entities
