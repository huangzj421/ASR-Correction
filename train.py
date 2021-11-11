# -*- coding: utf-8 -*-
"""
# #### PyTorch代码
# - [seq2seq-tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# - [Tutorial from Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
# - [IBM seq2seq](https://github.com/IBM/pytorch-seq2seq)
# - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
# - [text-generation](https://github.com/shibing624/text-generation)
"""

import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.append('..')
from bertseq2seq import config
from bertseq2seq.data_reader import load_bert_data
from bertseq2seq.utils.logger import logger
from bertseq2seq.seq2seq_model import Seq2SeqModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(arch, train_path, dev_path, batch_size, embed_size, hidden_size, dropout, epochs,
          model_dir, max_length, use_segment, model_name_or_path, evaluate_during_training, eval_batch_size, evaluate_during_training_steps):
    if arch == "bert":
        # Bert Seq2seq model
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": max_length if max_length else 128,
            "train_batch_size": batch_size if batch_size else 8,
            "num_train_epochs": epochs if epochs else 10,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "silent": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": evaluate_during_training,
            "evaluate_during_training_verbose": evaluate_during_training,
            "eval_batch_size": eval_batch_size if eval_batch_size else 64,
            "evaluate_during_training_steps": evaluate_during_training_steps if evaluate_during_training_steps else 2500,
            "use_multiprocessing": False,
            "save_best_model": True,
            "max_length": max_length if max_length else 128,  # The maximum length of the sequence to be generated.
            "output_dir": model_dir if model_dir else "output/bertseq2seq_demo/",
        }

        use_cuda = True if torch.cuda.is_available() else False
        # encoder_type=None, encoder_name=None, decoder_name=None
        # encoder_name="bert-base-chinese"
        model = Seq2SeqModel(arch, "{}".format(model_name_or_path), "{}".format(model_name_or_path), args=model_args, use_cuda=use_cuda)

        print('start train bertseq2seq ...')
        train_data = load_bert_data(train_path, use_segment)
        # train_data, dev_data = train_test_split(data, test_size=0.1, shuffle=True)
        dev_data = load_bert_data(dev_path, use_segment)

        train_df = pd.DataFrame(train_data, columns=['input_text', 'target_text'])
        dev_df = pd.DataFrame(dev_data, columns=['input_text', 'target_text'])

        model.train_model(train_df, eval_data=dev_df)
    else:
        logger.error('error arch: {}'.format(arch))
        raise ValueError("Model arch choose error. Must use one of seq2seq model.")


if __name__ == '__main__':
    print('device: %s' % device)
    train(config.arch,
          config.train_path,
          config.dev_path,
          config.batch_size,
          config.embed_size,
          config.hidden_size,
          config.dropout,
          config.epochs,
          config.model_dir,
          config.max_length,
          config.use_segment,
          config.model_name_or_path,
          config.evaluate_during_training,
          config.eval_batch_size,
          config.evaluate_during_training_steps
          )
