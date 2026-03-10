# -*- coding: utf-8 -*-
"""
将 prepare_keyword_correction_jsonl 生成的 train.jsonl / dev.jsonl 还原成可读文本，
用于检查训推一致性和训练集格式。

每条样本格式（与 qwen_utils.preprocess_one_correction_pair 一致）：
- input_ids: [system + user + assistant] 的完整 token 序列
- labels: 前面 prompt 部分为 -100（不参与 loss），后面 assistant 回复部分与 input_ids 一致（参与 loss）

用法:
  python scripts/inspect_keyword_correction_jsonl.py train.jsonl
  python scripts/inspect_keyword_correction_jsonl.py train.jsonl --model /path/to/model --max_samples 10
  python scripts/inspect_keyword_correction_jsonl.py train.jsonl --output decoded.txt --max_samples 100
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 可选：从 config 读 model_name_or_path
try:
    import config_corrector3 as config
    DEFAULT_MODEL = getattr(config, "model_name_or_path", "Qwen/Qwen3-4B")
except Exception:
    DEFAULT_MODEL = "Qwen/Qwen3-4B"

IGNORE_INDEX = -100


def get_prompt_len(labels: list) -> int:
    """返回第一个 labels[i] != IGNORE_INDEX 的下标，即 prompt 长度。若全为 -100 则返回 len(labels)。"""
    for i, lb in enumerate(labels):
        if lb != IGNORE_INDEX:
            return i
    return len(labels)


def main():
    parser = argparse.ArgumentParser(description="将 keyword correction jsonl 的 input_ids 解码为可读文本，检查格式")
    parser.add_argument("input", type=str, help="输入 jsonl 路径，如 train.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="tokenizer 所在模型路径（与 prepare 时一致）")
    parser.add_argument("--max_samples", type=int, default=5, help="最多解码并打印的样本数，0 表示全部")
    parser.add_argument("--output", type=str, default=None, help="若指定，将解码结果写入该文件；否则只打印到 stdout")
    parser.add_argument("--validate_only", action="store_true", help="只做格式校验（labels 长度、-100 边界、可解码），不输出解码文本")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    out_file = None
    if args.output and not args.validate_only:
        out_file = open(args.output, "w", encoding="utf-8")

    def emit(s: str = ""):
        if out_file:
            out_file.write(s + "\n")
        else:
            print(s)

    n = 0
    errors = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"行 {line_no}: JSON 解析失败: {e}")
                continue
            if "input_ids" not in obj or "labels" not in obj:
                errors.append(f"行 {line_no}: 缺少 input_ids 或 labels")
                continue
            input_ids = obj["input_ids"]
            labels = obj["labels"]
            if len(input_ids) != len(labels):
                errors.append(f"行 {line_no}: input_ids 长度 {len(input_ids)} != labels 长度 {len(labels)}")
                continue
            prompt_len = get_prompt_len(labels)
            # 校验：从 prompt_len 起 labels 应与 input_ids 一致
            for i in range(prompt_len, len(labels)):
                if labels[i] != input_ids[i] and labels[i] != IGNORE_INDEX:
                    errors.append(f"行 {line_no}: 位置 i={i} labels[i]={labels[i]} != input_ids[i]={input_ids[i]}")
                    break
            if args.validate_only:
                # 仅校验能否解码
                try:
                    tokenizer.decode(input_ids, skip_special_tokens=False)
                except Exception as e:
                    errors.append(f"行 {line_no}: 解码异常: {e}")
                n += 1
                continue
            if args.max_samples > 0 and n >= args.max_samples:
                n += 1
                continue
            # 解码
            import pdb;pdb.set_trace()
            prompt_ids = input_ids[:prompt_len]
            target_ids = input_ids[prompt_len:]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
            full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            emit(f"{'='*60}")
            emit(f"样本索引: {n}  (行号 {line_no})")
            emit(f"input_ids 长度: {len(input_ids)},  prompt 长度: {prompt_len},  目标长度: {len(target_ids)}")
            emit(f"{'-'*60}")
            emit("[不参与 loss — system + user + assistant 开头]")
            emit(prompt_text)
            emit(f"{'-'*60}")
            emit("[参与 loss — 模型需要预测的 assistant 回复]")
            emit(target_text)
            emit(f"{'-'*60}")
            emit("[完整序列解码]")
            emit(full_text)
            emit("")
            n += 1

    if out_file:
        out_file.close()

    if errors:
        emit("格式校验发现问题:")
        for e in errors:
            emit("  " + e)
    else:
        emit(f"格式校验通过。共处理样本数: {n}")
    if args.validate_only and not args.output:
        print(f"校验完成，共 {n} 条样本，错误 {len(errors)} 条")


if __name__ == "__main__":
    main()
