# -*- coding: utf-8 -*-
"""
使用 Hugging Face 上的 twnlp/ChineseErrorCorrector3-4B 对测试集做纠错。
输入格式与 run_infer.py 一致：每行为「序号 错误文本<mask_99>关键词<mask_99>」，
本脚本只使用「错误文本」部分调用 CEC3 模型（该模型不做关键词增强），
输出每行为「序号 纠错后的文本」。
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_reader import KEYWORD_SEP
from utils.logger import logger

# CEC3 官方 prompt（见 https://huggingface.co/twnlp/ChineseErrorCorrector3-4B）
CEC3_PROMPT = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："
CEC3_MODEL_NAME = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/hf_models/huggingface.co/twnlp/ChineseErrorCorrector3-4B"


def parse_infer_line(line):
    """
    解析一行：序号 错误文本<mask_99>关键词<mask_99>
    返回 (idx_str, error_text) 或 None。
    """
    line = line.strip()
    if not line:
        return None
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
    return (idx_str, error_text)


def load_input_lines(path):
    """加载输入文件，返回 [(idx_str, error_text), ...]。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_infer_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def build_messages(text):
    """单条输入构造 chat messages。"""
    return [{"role": "user", "content": CEC3_PROMPT + text}]


def run():
    parser = argparse.ArgumentParser(
        description="使用 ChineseErrorCorrector3-4B 对 input_txt 纠错，输出 序号 纠错文本"
    )
    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help="输入 txt，每行: 序号 错误文本<mask_99>关键词<mask_99>",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="output/corrector3_ori.txt",
        help="输出 txt，默认 input_txt 同目录下的 <input_basename>_cec3.txt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=CEC3_MODEL_NAME,
        help="Hugging Face 模型名或本地路径，默认 %s" % CEC3_MODEL_NAME,
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--enable_thinking", action="store_true", help="是否开启 thinking 模式（默认关闭以加快推理）")
    args = parser.parse_args()

    rows = load_input_lines(args.input_txt)
    if not rows:
        logger.warning("未解析到任何有效行，请检查输入格式：序号 错误文本<mask_99>关键词<mask_99>")
        return
    indices = [r[0] for r in rows]
    error_texts = [r[1] for r in rows]
    logger.info("输入行数: %d", len(rows))

    if args.output_txt is None:
        base = os.path.splitext(os.path.basename(args.input_txt))[0]
        out_dir = os.path.dirname(os.path.abspath(args.input_txt))
        args.output_txt = os.path.join(out_dir, base + "_cec3.txt")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)) or ".", exist_ok=True)

    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("加载模型: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_preds = []
    batch_size = args.batch_size
    for start in range(0, len(error_texts), batch_size):
        batch_texts = error_texts[start : start + batch_size]
        messages_batch = [build_messages(t) for t in batch_texts]
        # apply_chat_template 对每条单独得到 prompt 再 tokenize，便于 padding
        texts_for_model = [
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msg in messages_batch
        ]
        model_inputs = tokenizer(
            texts_for_model,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        if device_map:
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")

        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 只解码新生成部分
        for i in range(len(batch_texts)):
            input_len = input_ids[i].shape[0]
            if attention_mask is not None:
                input_len = attention_mask[i].sum().item()
            out_ids = generated[i][input_len:]
            pred = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            all_preds.append(pred)
        logger.info("已推理 %d / %d", min(start + batch_size, len(error_texts)), len(error_texts))
        # import pdb; pdb.set_trace()

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for idx, pred in zip(indices, all_preds):
            f.write("{} {}\n".format(idx, pred))
    logger.info("输出已写入: %s", args.output_txt)


if __name__ == "__main__":
    run()
