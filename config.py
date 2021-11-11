# -*- coding: utf-8 -*-

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
sighan_train_path = os.path.join(pwd_path, './data/sighan_2015/train.tsv')
sighan_dev_path = os.path.join(pwd_path, './data/sighan_2015/dev.tsv')
speechx_train_path = os.path.join(pwd_path, './data/mandarin-accented/train.csv')
speechx_dev_path = os.path.join(pwd_path, './data/mandarin-accented/valid.csv')

use_segment = True
segment_type = 'char'  # 'word' use jieba.lcut; 'char' use list(sentence)

dataset = 'manacc'  # 'sighan' or 'aishell1' or 'magicdata' or 'mandarin'
output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train_{}.txt'.format(dataset))
# Validation data path.
dev_path = os.path.join(output_dir, 'dev_{}.txt'.format(dataset))

arch = "bert"  # 'bertseq2seq'
model_name_or_path = "bert-base-chinese"  # for "bert-base-chinese"

# config
model_dir = os.path.join(output_dir, 'model_{}_{}'.format(arch, dataset))

batch_size = 16
epochs = 10
max_length = 128
evaluate_during_training = False
eval_batch_size = 16
evaluate_during_training_steps = 2500

gpu_id = 0
dropout = 0.25
embed_size = 128
hidden_size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
