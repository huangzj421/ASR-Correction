# -*- coding: utf-8 -*-
"""
Qwen3 ASR 纠错 Demo：预处理 → 训练 → 预测（已移除 BERTSeq2Seq，仅保留 Qwen3）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train
from infer import Inference
from preprocess import get_data_file, get_data_filex, save_corpus_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_train_path", default="./data/sighan_2015/train.tsv", type=str)
    parser.add_argument("--raw_dev_path", default="./data/sighan_2015/dev.tsv", type=str)
    parser.add_argument("--dataset", default="sighan", type=str, choices=["sighan", "cged"])
    parser.add_argument("--no_segment", action="store_true", help="Do not segment in preprocess")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--segment_type", default="char", type=str, choices=["char", "word"])
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B", type=str)
    parser.add_argument("--model_dir", default="output/qwen3_demo/", type=str)
    parser.add_argument("--train_path", default="output/train_demo.txt", type=str)
    parser.add_argument("--dev_path", default="output/dev_demo.txt", type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--save_steps", default=400, type=int)
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.do_train:
        use_segment = not args.no_segment
        if args.dataset == "sighan":
            data_train = get_data_file(args.raw_train_path, use_segment, args.segment_type)
            data_dev = get_data_file(args.raw_dev_path, use_segment, args.segment_type)
        else:
            data_train = get_data_filex(args.raw_train_path, use_segment, args.segment_type)
            data_dev = get_data_filex(args.raw_dev_path, use_segment, args.segment_type)
        save_corpus_data(data_train, data_dev, args.train_path, args.dev_path)
        train(
            args.train_path,
            args.dev_path,
            args.batch_size,
            args.epochs,
            args.model_dir,
            args.max_length,
            args.max_seq_length,
            use_segment,
            args.model_name_or_path,
            args.use_peft,
            args.learning_rate,
            args.eval_steps,
            args.save_steps,
        )

    if args.do_predict:
        inference = Inference(args.model_dir, max_length=args.max_length)
        inputs = [
            "老是较书。",
            "感谢等五分以后，碰到一位很棒的奴生跟我可聊。",
            "遇到一位很棒的奴生跟我聊天。",
            "遇到一位很美的女生跟我疗天。",
            "他们只能有两个选择：接受降新或自动离职。",
            "王天华开心得一直说话。",
        ]
        outputs = inference.predict(inputs)
        for a, b in zip(inputs, outputs):
            print("input  :", a)
            print("predict:", b)
            print()


if __name__ == "__main__":
    main()
