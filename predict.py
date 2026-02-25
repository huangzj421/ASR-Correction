# -*- coding: utf-8 -*-
"""Qwen3 纠错交互预测（已移除 BERTSeq2Seq）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from infer import Inference


if __name__ == "__main__":
    m = Inference(
        config.model_dir,
        max_length=config.max_length,
        eval_batch_size=getattr(config, "eval_batch_size", 16),
    )
    print("Qwen3 纠错已加载，输入文本后回车得到纠错结果，输入 Tab 结束")
    while True:
        try:
            inputs = input("输入文本：").strip()
        except EOFError:
            break
        if inputs == "\t":
            break
        if not inputs:
            continue
        outputs = m.predict([inputs])
        print("纠错结果：" + outputs[0])
        print()
