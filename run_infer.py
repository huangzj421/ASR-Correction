# -*- coding: utf-8 -*-
"""
对关键词纠错模型的所有 checkpoint 做批量推理：输入 txt 每行为「序号 错误文本<mask_99>关键词<mask_99>」，
对每个 checkpoint 输出一个 txt，每行为「序号 纠错后的文本」。
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_reader import KEYWORD_SEP, build_keyword_input
from infer import Inference
from utils.logger import logger


def parse_infer_line(line):
    """
    解析一行：序号 错误文本<mask_99>关键词<mask_99>
    返回 (idx_str, input_text) 或 None（解析失败）。
    """
    line = line.strip()
    if not line:
        return None
    # 第一个空格分隔序号与内容
    parts = line.split(" ", 1)
    if len(parts) != 2:
        return None
    idx_str, rest = parts[0].strip(), parts[1].strip()
    if KEYWORD_SEP not in rest:
        return None
    segs = rest.split(KEYWORD_SEP)
    if len(segs) < 2:
        return None
    error_text = segs[0].strip()
    keywords_str = segs[1].strip()
    input_text = build_keyword_input(error_text, keywords_str)
    return (idx_str, input_text)


def load_input_lines(path):
    """加载输入文件，返回 [(idx_str, input_text), ...]。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_infer_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def list_checkpoints(model_dir):
    """
    列出所有 checkpoint：checkpoint-* 子目录（按步数排序），
    若根目录也有 adapter_config.json 则作为 "final" 放在最后。
    """
    checkpoints = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("checkpoint-"):
            adapter = os.path.join(path, "adapter_config.json")
            if os.path.isfile(adapter):
                checkpoints.append((name, path))
    def step_key(x):
        m = re.search(r"checkpoint-(\d+)", x[0])
        return int(m.group(1)) if m else 0
    checkpoints.sort(key=step_key)
    root_adapter = os.path.join(model_dir, "adapter_config.json")
    if os.path.isfile(root_adapter):
        checkpoints.append(("final", model_dir))
    return checkpoints


def run():
    parser = argparse.ArgumentParser(description="对每个 checkpoint 推理并输出 序号 纠错文本")
    parser.add_argument("--input_txt", type=str, required=True, help="输入 txt，每行: 序号 错误文本<mask_99>关键词<mask_99>")
    parser.add_argument("--model_dir", type=str, default=None, help="模型目录（含 checkpoint-* 的目录），默认 config.keyword_model_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认 model_dir 下的 infer_output")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_dir = args.model_dir or getattr(config, "keyword_model_dir", "output/model_qwen3_keyword")
    output_dir = args.output_dir or os.path.join(model_dir, "infer_output")
    os.makedirs(output_dir, exist_ok=True)

    rows = load_input_lines(args.input_txt)
    if not rows:
        logger.warning("未解析到任何有效行，请检查输入格式：序号 错误文本<mask_99>关键词<mask_99>")
        return
    indices = [r[0] for r in rows]
    input_texts = [r[1] for r in rows]
    logger.info("输入行数: %d", len(rows))

    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        logger.warning("未找到任何 checkpoint（需含 adapter_config.json），尝试将 model_dir 视为单个 checkpoint")
        checkpoints = [("model", model_dir)]

    for ckpt_name, ckpt_path in checkpoints:
        out_path = os.path.join(output_dir, "pred_{}.txt".format(ckpt_name))
        logger.info("Checkpoint %s -> %s", ckpt_name, out_path)
        try:
            # 基座名从 model_dir 的 model_args.json 读，PEFT 从 ckpt_path 加载（final 时 ckpt_path==model_dir）
            inf = Inference(
                model_dir,
                max_length=args.max_length,
                eval_batch_size=args.batch_size,
                checkpoint_path=ckpt_path,
            )
            preds = inf.model.predict(input_texts, max_length=args.max_length)
            with open(out_path, "w", encoding="utf-8") as f:
                for idx, pred in zip(indices, preds):
                    f.write("{} {}\n".format(idx, pred.strip()))
        except Exception as e:
            logger.exception("Checkpoint %s 推理失败: %s", ckpt_name, e)
            continue
    logger.info("全部完成，输出目录: %s", output_dir)


if __name__ == "__main__":
    run()
