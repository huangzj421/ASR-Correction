# -*- coding: utf-8 -*-
"""
Qwen3 ASR 文本纠错训练入口（已移除 BERTSeq2Seq，仅保留 Qwen3）
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_reader import load_bert_data
from utils.logger import logger
from qwen_model import QwenCorrectionModel


def train(
    train_path,
    dev_path,
    batch_size,
    epochs,
    model_dir,
    max_length,
    max_seq_length,
    use_segment,
    model_name_or_path,
    use_peft,
    learning_rate,
    eval_steps,
    save_steps,
):
    use_cuda = torch.cuda.is_available()
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": max_seq_length or 256,
        "max_length": max_length or 256,
        "per_device_train_batch_size": batch_size or 4,
        "num_train_epochs": epochs or 3,
        "output_dir": model_dir or "output/qwen3_demo/",
        "use_peft": use_peft if use_peft is not None else True,
        "eval_steps": eval_steps or 200,
        "save_steps": save_steps or 400,
        "learning_rate": learning_rate or 2e-5,
        "bf16": use_cuda,
        "prompt_template_name": "qwen",
    }
    model = QwenCorrectionModel(model_name_or_path, args=model_args, use_cuda=use_cuda)
    train_data = load_bert_data(train_path, use_segment)
    dev_data = load_bert_data(dev_path, use_segment)
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
    dev_df = pd.DataFrame(dev_data, columns=["input_text", "target_text"])
    logger.info("Train samples: %d, dev samples: %d", len(train_df), len(dev_df))
    model.train_model(train_df, eval_data=dev_df)


if __name__ == "__main__":
    train(
        config.train_path,
        config.dev_path,
        config.batch_size,
        config.epochs,
        config.model_dir,
        config.max_length,
        getattr(config, "max_seq_length", config.max_length),
        config.use_segment,
        config.model_name_or_path,
        getattr(config, "use_peft", True),
        getattr(config, "learning_rate", 2e-5),
        getattr(config, "eval_steps", 200),
        getattr(config, "save_steps", 400),
    )
