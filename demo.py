# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.append("..")
from bertseq2seq.train import train, device
from bertseq2seq.infer import Inference
from bertseq2seq.preprocess import get_data_file, get_data_filex, save_corpus_data


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--raw_train_path",
                        default="./data/sighan_2015/train.tsv", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
                        )
    parser.add_argument("--raw_dev_path",
                        default="./data/sighan_2015/dev.tsv", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
                        )
    parser.add_argument("--dataset", default="sighan", type=str,
                        help="Dataset name. selected in the list:" + ", ".join(["sighan", "cged"])
                        )
    parser.add_argument("--no_segment", action="store_true", help="Whether not to segment train data in preprocess")
    parser.add_argument("--do_train", action="store_true", help="Whether not to train")
    parser.add_argument("--do_predict", action="store_true", help="Whether not to predict")
    parser.add_argument("--segment_type", default="char", type=str,
                        help="Segment data type, selected in list: " + ", ".join(["char", "word"]))
    parser.add_argument("--model_name_or_path",
                        default="bert-base-chinese", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models",
                        )
    parser.add_argument("--model_dir", default="output/bertseq2seq_demo/", type=str, help="Dir for model save.")
    parser.add_argument("--arch",
                        default="bert", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(
                            ['seq2seq', 'convseq2seq', 'bertseq2seq']),
                        )
    parser.add_argument("--train_path", default="output/train_demo.txt", type=str, help="Train file after preprocess.")
    parser.add_argument("--dev_path", default="output/dev_demo.txt", type=str, help="Dev file after preprocess.")

    # Other parameters
    parser.add_argument("--max_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, sequences shorter padded.",
                        )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--embed_size", default=128, type=int, help="Embedding size.")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size.")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=10, type=int, help="Epoch num.")
    parser.add_argument("--evaluate_during_training", default=False, type=bool, help="Whether to evaluate.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size of evaluation.")
    parser.add_argument("--evaluate_during_training_steps", default=2500, type=int, help="Evaluate every training steps.")

    args = parser.parse_args()
    print(args)

    # Preprocess
    os.makedirs(args.model_dir, exist_ok=True)

    # Train
    if args.do_train:
        # Preprocess
        args.use_segment = False if args.no_segment else True
        data_list_train = []
        data_list_dev = []
        if args.dataset == 'sighan':
            data_train = get_data_file(args.raw_train_path, args.use_segment, args.segment_type)
            data_dev = get_data_file(args.raw_dev_path, args.use_segment, args.segment_type)
        else:
            data_train = get_data_filex(args.raw_train_path, args.use_segment, args.segment_type)
            data_dev = get_data_filex(args.raw_dev_path, args.use_segment, args.segment_type)
        data_list_train.extend(data_train)
        data_list_dev.extend(data_dev)
        save_corpus_data(data_list_train, data_list_dev, args.train_path, args.dev_path)
        # Train model with train data file
        train(args.arch,
              args.train_path,
              args.dev_path,
              args.batch_size,
              args.embed_size,
              args.hidden_size,
              args.dropout,
              args.epochs,
              args.model_dir,
              args.max_length,
              args.use_segment,
              args.model_name_or_path,
              args.evaluate_during_training,
              args.eval_batch_size,
              args.evaluate_during_training_steps
              )

    # Predict
    if args.do_predict:
        inference = Inference(args.arch,
                              args.model_dir,
                              embed_size=args.embed_size,
                              hidden_size=args.hidden_size,
                              dropout=args.dropout,
                              max_length=args.max_length,
                              evaluate_during_training=args.evaluate_during_training,
                              eval_batch_size=args.eval_batch_size,
                              evaluate_during_training_steps=args.evaluate_during_training_steps
                              )
        inputs = [
            '老是较书。',
            '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
            '遇到一位很棒的奴生跟我聊天。',
            '遇到一位很美的女生跟我疗天。',
            '他们只能有两个选择：接受降新或自动离职。',
            '王天华开心得一直说话。'
        ]
        outputs = inference.predict(inputs)
        for a, b in zip(inputs, outputs):
            print('input  :', a)
            print('predict:', b)
            print()


if __name__ == "__main__":
    main()
