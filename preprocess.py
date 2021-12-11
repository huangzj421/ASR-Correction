# -*- coding: utf-8 -*-

import os
import sys
from codecs import open
from xml.dom import minidom

from sklearn.model_selection import train_test_split

sys.path.append('..')
from utils.tokenizer import segment
import config

def get_data_file(path, use_segment, segment_type):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            source = ' '.join(segment(parts[0].strip(), cut_type=segment_type)) if use_segment else parts[0].strip()
            target = ' '.join(segment(parts[1].strip(), cut_type=segment_type)) if use_segment else parts[1].strip()

            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def get_data_filex(path, use_segment, segment_type):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("recognition"):
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            source = ' '.join(segment(parts[0].strip(), cut_type=segment_type)) if use_segment else parts[0].strip()
            target = ' '.join(segment(parts[1].strip(), cut_type=segment_type)) if use_segment else parts[1].strip()

            pair = [source, target]
            print(pair)
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, data_path):
    dirname = os.path.dirname(data_path)
    os.makedirs(dirname, exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write(src + '\t' + dst + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def save_corpus_data(data_list_train, data_list_dev, train_data_path, dev_data_path):
    _save_data(data_list_train, train_data_path)
    _save_data(data_list_dev, dev_data_path)


if __name__ == '__main__':
    # train data
    data_list_train = []
    data_list_dev = []
    if config.dataset == 'sighan':
        data_train = get_data_file(config.sighan_train_path, config.use_segment, config.segment_type)
        data_dev = get_data_file(config.sighan_dev_path, config.use_segment, config.segment_type)
    else:
        data_train = get_data_filex(config.speechx_train_path, config.use_segment, config.segment_type)
        data_dev = get_data_filex(config.speechx_dev_path, config.use_segment, config.segment_type)
    data_list_train.extend(data_train)
    data_list_dev.extend(data_dev)
    save_corpus_data(data_list_train, data_list_dev, config.train_path, config.dev_path)
