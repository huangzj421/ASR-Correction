# -*- coding:utf-8 -*-
"""
@description: 
"""
import os
import sys

import numpy as np
import torch
from jiwer import wer
import sacrebleu
sys.path.append('..')

import config
from data_reader import load_word_dict
from seq2seq_model import Seq2SeqModel
from utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference(object):
    def __init__(self, arch, model_dir, embed_size=50, hidden_size=50, dropout=0.5, max_length=128, 
        batch_size=8, epochs=10, evaluate_during_training=True, eval_batch_size=64, evaluate_during_training_steps=2500):
        logger.debug("device: {}".format(device))
        if arch == "bert":
            # Bert Seq2seq model
            logger.debug('use bert seq2seq model.')
            use_cuda = True if torch.cuda.is_available() else False
            
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
                "max_length": max_length if max_length else 128,  # The maximum length of the sequence
                "output_dir": model_dir if model_dir else "output/bertseq2seq_demo/",
            }
            # encoder_type=None, encoder_name=None, decoder_name=None
            self.model = Seq2SeqModel(arch, "{}/encoder".format(model_dir),
                                      "{}/decoder".format(model_dir), args=model_args, use_cuda=use_cuda)
        else:
            logger.error('error arch: {}'.format(arch))
            raise ValueError("Model arch choose error. Must use one of seq2seq model.")
        self.arch = arch
        self.max_length = max_length

    def predict(self, sentence_list):
        result = []
        if self.arch == "bert":
            corrected_sents = self.model.predict(sentence_list)
            result = [i.replace(' ', '') for i in corrected_sents]
        else:
            raise ValueError('error arch.')
        return result


if __name__ == "__main__":
    m = Inference(config.arch,
                  config.model_dir,
                  embed_size=config.embed_size,
                  hidden_size=config.hidden_size,
                  dropout=config.dropout,
                  max_length=config.max_length,
                  batch_size=config.batch_size,
                  epochs=config.epochs,
                  evaluate_during_training=config.evaluate_during_training,
                  eval_batch_size=config.eval_batch_size,
                  evaluate_during_training_steps=config.evaluate_during_training_steps
                  )
    """
    inputs = [
        '??? ??? ??? ??????',
        '??? ??? ??? ??? ??? ??? ????????????????????????????????????????????????',
        '??????????????????????????????????????????',
        '??????????????????????????????????????????',
        '????????????????????????????????????????????????????????????',
        '?????????????????????????????????'
    ]
    outputs = m.predict(inputs)
    with open('output.txt','a') as f:
        for a, b in zip(inputs, outputs):
            f.write(a + '\n')
            f.write(b + '\n')
            f.write('\n')
    """
    source_list = []
    target_list = []
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(pwd_path, './data/mandarin-accented/test.csv')
    save_path = os.path.join(pwd_path, 'output_manacc.txt')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("recognition"):
                continue
            parts = line.split(",")
            #if len(parts) != 2:
            #    continue
            source_list.append(parts[0])
            target_list.append(parts[1])
        
        CER_src = 0
        CER_corr = 0
        count = 0
        corrected_source_list = m.predict(source_list)
        BLEU_src = sacrebleu.corpus_bleu(source_list, [target_list], tokenize='zh').score
        BLEU_corr = sacrebleu.corpus_bleu(corrected_source_list, [target_list], tokenize='zh').score
        with open(save_path,'a') as f:
            for source, corrected_source, target in zip(source_list, corrected_source_list, target_list):        
                #corrected_length = len(source)-4*source.count('<unk>')
                corrected_source_b = ' '.join(list(corrected_source.strip()))
                source_b = ' '.join(list(source.strip()))
                target_b = ' '.join(list(target.strip()))

                CER_src += wer(target_b, source_b)
                CER_corr += wer(target_b, corrected_source_b)
                count += 1
                
                f.write("\n")     
                f.write("src:{}".format(source))
                f.write("\n")
                f.write("corr:{}".format(corrected_source))
                f.write("\n")
                f.write("tgt:{}".format(target))
                f.write("\n")
            
            f.write("\n")
            f.write("BLEU_src:{} => BLEU_corr:{}".format(BLEU_src, BLEU_corr))
            f.write("\n")
            f.write("CER_src:{} => CER_corr:{}".format(CER_src/count, CER_corr/count))
        
# result:
# input:?????????????????????
# output:???????????????
# input:???????????????????????????
# output:????????????????????????
# input:?????????????????????
