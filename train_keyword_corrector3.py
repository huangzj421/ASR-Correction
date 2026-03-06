# -*- coding: utf-8 -*-
"""
上下文关键词纠错训练：正例（正确词在关键词里）+ 负例（全干扰，希望不纠错），
数据格式 错误文本<mask_99>关键词1, 关键词2, ...<mask_99>正确文本

大数据量时请先用 scripts/prepare_keyword_correction_jsonl.py 预处理成 jsonl+filelist，
再传 --train_filelist / --eval_filelist 流式训练，避免全量进内存。
"""
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_corrector3 as config
from data_reader import load_keyword_correction_data
from utils.logger import logger
from qwen_model import QwenCorrectionModel


def train_keyword(
    pos_dir=None,
    neg_dir=None,
    neg_ratio=1.0,
    dev_ratio=0.1,
    seed=42,
    batch_size=None,
    epochs=None,
    model_dir=None,
    max_length=None,
    max_seq_length=None,
    model_name_or_path=None,
    use_peft=True,
    learning_rate=None,
    eval_steps=None,
    save_steps=None,
    gradient_accumulation_steps=None,
    train_filelist=None,
    eval_filelist=None,
    resume_from_checkpoint=None,
):
    """
    使用关键词纠错数据训练。
    - 若提供 train_filelist / eval_filelist：流式加载（不把全量数据读入内存）。
    - 否则用 pos_dir + neg_dir 加载原始文本，全量在内存，适合小数据。
    """
    use_cuda = torch.cuda.is_available()
    model_dir = model_dir or getattr(config, "keyword_model_dir", "output/model_qwen3_keyword")
    model_name_or_path = model_name_or_path or config.model_name_or_path
    batch_size = batch_size or getattr(config, "batch_size", 4)
    epochs = epochs or getattr(config, "epochs", 3)
    max_length = max_length or getattr(config, "max_length", 256)
    max_seq_length = max_seq_length or getattr(config, "max_seq_length", 256)
    learning_rate = learning_rate or getattr(config, "learning_rate", 2e-5)
    eval_steps = eval_steps or getattr(config, "eval_steps", 200)
    save_steps = save_steps or getattr(config, "save_steps", 400)
    gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else getattr(config, "gradient_accumulation_steps", 1)

    if train_filelist and os.path.isfile(train_filelist):
        train_data = train_filelist
        eval_data = eval_filelist if (eval_filelist and os.path.isfile(eval_filelist)) else None
        logger.info(
            "Keyword correction: 流式加载 train_filelist=%s, eval_filelist=%s",
            train_filelist, eval_filelist,
        )
    else:
        pos_dir = pos_dir or getattr(config, "keyword_pos_dir", "data/keyword/pos")
        neg_dir = neg_dir or getattr(config, "keyword_neg_dir", "data/keyword/neg")
        rows = load_keyword_correction_data(pos_dir, neg_dir, neg_ratio=neg_ratio, seed=seed)
        if not rows:
            raise ValueError("未读到任何数据，请检查 pos_dir / neg_dir 及 <mask_99> 格式。")
        train_rows, dev_rows = train_test_split(rows, test_size=dev_ratio, random_state=seed, shuffle=True)
        train_data = pd.DataFrame(train_rows, columns=["input_text", "target_text"])
        eval_data = pd.DataFrame(dev_rows, columns=["input_text", "target_text"])
        logger.info(
            "Keyword correction: train=%d, dev=%d (pos_dir=%s, neg_ratio=%s)",
            len(train_data), len(eval_data), pos_dir, neg_ratio,
        )

    resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint is not None else getattr(config, "resume_from_checkpoint", None)
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": not resume_from_checkpoint,
        "max_seq_length": max_seq_length,
        "max_length": max_length,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": epochs,
        "output_dir": model_dir,
        "use_peft": use_peft,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "bf16": use_cuda,
        "prompt_template_name": "qwen",
    }
    if resume_from_checkpoint is not None:
        model_args["resume_from_checkpoint"] = resume_from_checkpoint
    model = QwenCorrectionModel(model_name_or_path, args=model_args, use_cuda=use_cuda)
    model.train_model(train_data, eval_data=eval_data)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="关键词纠错训练（支持 filelist 流式）")
    p.add_argument("--pos_dir", type=str, default=None)
    p.add_argument("--neg_dir", type=str, default=None)
    p.add_argument("--neg_ratio", type=float, default=None)
    p.add_argument("--dev_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_filelist", type=str, default=None, help="训练集 filelist，与 eval_filelist 一起用时为流式加载")
    p.add_argument("--eval_filelist", type=str, default=None, help="验证集 filelist")
    p.add_argument("--model_dir", type=str, default=None,
                   help="模型输出目录（训练 checkpoint 与最终保存路径），不传则用 config 的 keyword_model_dir")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None, help="梯度累积步数，增大可等效更大 batch、减少 step 数")
    p.add_argument("--resume", type=str, default=None, nargs="?", const="True",
                   help="断点续训：不传值时从 output_dir 中找最新 checkpoint；传路径则从该 checkpoint 目录恢复，如 --resume output/model_qwen3_keyword/checkpoint-800")
    a = p.parse_args()

    resume_from_checkpoint = None
    if a.resume is not None:
        if a.resume == "True" or a.resume == "":
            resume_from_checkpoint = True  # Trainer 会在 output_dir 中自动找最新 checkpoint
        else:
            resume_from_checkpoint = a.resume

    train_keyword(
        pos_dir=a.pos_dir,
        neg_dir=a.neg_dir,
        neg_ratio=a.neg_ratio or getattr(config, "keyword_neg_ratio", 1.0),
        dev_ratio=a.dev_ratio if a.dev_ratio is not None else getattr(config, "keyword_dev_ratio", 0.1),
        seed=a.seed if a.seed is not None else getattr(config, "manual_seed", 42),
        train_filelist=a.train_filelist,
        eval_filelist=a.eval_filelist,
        model_dir=a.model_dir,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
