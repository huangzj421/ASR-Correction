# -*- coding: utf-8 -*-
"""
上下文关键词纠错训练：正例（正确词在关键词里）+ 负例（全干扰，希望不纠错），
数据格式 错误文本<mask_99>关键词1, 关键词2, ...<mask_99>正确文本
"""
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_reader import load_keyword_correction_data
from utils.logger import logger
from qwen_model import QwenCorrectionModel


def train_keyword(
    pos_dir,
    neg_dir,
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
):
    """使用关键词纠错数据训练：正例目录 + 负例目录，负例按 neg_ratio 倍正例数量采样后打乱。"""
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

    rows = load_keyword_correction_data(pos_dir, neg_dir, neg_ratio=neg_ratio, seed=seed)
    if not rows:
        raise ValueError("未读到任何数据，请检查 pos_dir / neg_dir 及 <mask_99> 格式。")
    train_rows, dev_rows = train_test_split(rows, test_size=dev_ratio, random_state=seed, shuffle=True)
    train_df = pd.DataFrame(train_rows, columns=["input_text", "target_text"])
    dev_df = pd.DataFrame(dev_rows, columns=["input_text", "target_text"])
    logger.info(
        "Keyword correction: train=%d, dev=%d (pos_dir=%s, neg_ratio=%s)",
        len(train_df), len(dev_df), pos_dir, neg_ratio,
    )

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": max_seq_length,
        "max_length": max_length,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": epochs,
        "output_dir": model_dir,
        "use_peft": use_peft,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "learning_rate": learning_rate,
        "bf16": use_cuda,
        "prompt_template_name": "qwen",
    }
    model = QwenCorrectionModel(model_name_or_path, args=model_args, use_cuda=use_cuda)
    model.train_model(train_df, eval_data=dev_df)


if __name__ == "__main__":
    pos_dir = getattr(config, "keyword_pos_dir", "data/keyword/pos")
    neg_dir = getattr(config, "keyword_neg_dir", "data/keyword/neg")
    neg_ratio = getattr(config, "keyword_neg_ratio", 1.0)
    train_keyword(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        neg_ratio=neg_ratio,
        dev_ratio=getattr(config, "keyword_dev_ratio", 0.1),
        seed=getattr(config, "manual_seed", 42),
    )
