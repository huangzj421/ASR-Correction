# -*- coding: utf-8 -*-
"""
流式 + 多进程（参考 merge_npy_chunk）：
1) 流式写 temp（一行 buffer），同时记录每行 offset，不把全量载入内存；
2) 只保留打乱索引 inv_perm；
3) 按 chunk（dev + 各 train shard）分片，多进程读 temp 按 offset tokenize 写出。

用法:
  python scripts/prepare_keyword_correction_jsonl.py --pos_dir ... --neg_dir ... --out_dir ...
  python scripts/prepare_keyword_correction_jsonl.py ... -j 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config_corrector3 as config
from data_reader import stream_pos_rows, reservoir_sample_neg_rows
from utils.logger import logger
from tqdm import tqdm

TEMP_RAW = ".prepare_keyword_raw.tmp"
TEMP_OFFSETS = ".prepare_keyword_offsets.npy"


def _write_sample_binary(f_out, input_ids: list, labels: list, dtype=np.int32):
    """写一条样本为二进制：4 字节 len_i + input_ids + 4 字节 len_l + labels。"""
    arr = np.array(input_ids, dtype=dtype)
    f_out.write(np.int32(len(arr)).tobytes())
    f_out.write(arr.tobytes())
    arr = np.array(labels, dtype=dtype)
    f_out.write(np.int32(len(arr)).tobytes())
    f_out.write(arr.tobytes())


def _tokenize_one_chunk(args_tuple: tuple) -> tuple[str, int]:
    """
    子进程：读 temp 中指定 (g, pos) 列表对应的行，tokenize 后按 pos 顺序写出到 out_path。
    args_tuple: (..., out_path, temp_path, offsets_path, ..., use_binary)
    chunk_list: list of (g, local_pos), 已按 local_pos 排序。
    use_binary: 若 True 写出 .bin + .bin.offsets（int32 序列，省空间），否则 jsonl。
    返回 (out_path 绝对路径, 写入条数)。
    """
    (
        chunk_id,
        chunk_list,
        out_path,
        temp_path,
        offsets_path,
        model_name_or_path,
        max_seq_length,
        max_length,
        use_binary,
    ) = args_tuple
    if not chunk_list:
        return (os.path.abspath(out_path), 0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoTokenizer
    from qwen_utils import QwenArgs, preprocess_one_correction_pair

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    qwen_args = QwenArgs(max_seq_length=max_seq_length, max_length=max_length)
    offsets = np.load(offsets_path, allow_pickle=False)
    N = len(offsets) - 1
    n_written = 0
    if use_binary:
        bin_path = out_path if out_path.endswith(".bin") else (out_path.replace(".jsonl", ".bin"))
        offsets_out = []
        with open(temp_path, "rb") as f, open(bin_path, "wb") as out:
            chunk_offsets = [0]
            for g, _ in chunk_list:
                start = int(offsets[g])
                end = int(offsets[g + 1])
                f.seek(start)
                raw = f.read(end - start).decode("utf-8")
                obj = json.loads(raw)
                input_text, target_text = obj[0], obj[1]
                one = preprocess_one_correction_pair(input_text, target_text, tokenizer, qwen_args)
                pos = out.tell()
                chunk_offsets.append(pos)
                _write_sample_binary(out, one["input_ids"], one["labels"])
                n_written += 1
            chunk_offsets.append(out.tell())
        np.save(bin_path + ".offsets.npy", np.array(chunk_offsets, dtype=np.int64), allow_pickle=False)
        return (os.path.abspath(bin_path), n_written)
    with open(temp_path, "rb") as f, open(out_path, "w", encoding="utf-8") as out:
        for g, _ in chunk_list:
            start = int(offsets[g])
            end = int(offsets[g + 1])
            f.seek(start)
            raw = f.read(end - start).decode("utf-8")
            obj = json.loads(raw)
            input_text, target_text = obj[0], obj[1]
            one = preprocess_one_correction_pair(input_text, target_text, tokenizer, qwen_args)
            out.write(json.dumps({"input_ids": one["input_ids"], "labels": one["labels"]}, ensure_ascii=False) + "\n")
            n_written += 1
    return (os.path.abspath(out_path), n_written)


def main():
    parser = argparse.ArgumentParser(description="预处理关键词纠错数据为 jsonl + filelist（流式+多进程）")
    parser.add_argument("--pos_dir", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_dense2b_txt/raw_data_itn/positive", help="正例目录")
    parser.add_argument("--neg_dir", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_dense2b_txt/raw_data_itn/negative", help="负例目录")
    parser.add_argument("--neg_ratio", type=float, default=None)
    parser.add_argument("--dev_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_dense2b_txt/qwen3_sft_itn", help="输出目录（jsonl 与 filelist）")
    parser.add_argument("--lines_per_shard", type=int, default=50000)
    parser.add_argument("-j", "--workers", type=int, default=30)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--format", type=str, choices=["jsonl", "binary"], default="jsonl",
                        help="jsonl=文本行；binary=int32 二进制，更省空间，需 Dataset 支持")
    args = parser.parse_args()

    pos_dir = args.pos_dir or getattr(config, "keyword_pos_dir", "data/keyword/pos")
    neg_dir = args.neg_dir or getattr(config, "keyword_neg_dir", "data/keyword/neg")
    neg_ratio = args.neg_ratio if args.neg_ratio is not None else getattr(config, "keyword_neg_ratio", 1.0)
    dev_ratio = args.dev_ratio if args.dev_ratio is not None else getattr(config, "keyword_dev_ratio", 0.1)
    seed = args.seed if args.seed is not None else getattr(config, "manual_seed", 42)
    out_dir = args.out_dir or os.path.join(config.output_dir, "keyword_corrector3_jsonl")
    lines_per_shard = args.lines_per_shard
    workers = max(1, args.workers)
    use_binary = args.format == "binary"
    ext = ".bin" if use_binary else ".jsonl"

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    temp_path = os.path.abspath(os.path.join(out_dir, TEMP_RAW))
    offsets_path = os.path.abspath(os.path.join(out_dir, TEMP_OFFSETS))

    model_name_or_path = getattr(config, "model_name_or_path", None)
    if not model_name_or_path:
        raise ValueError("config 中需设置 model_name_or_path")
    max_seq_length = args.max_seq_length or getattr(config, "max_seq_length", 512)
    max_length = args.max_length or getattr(config, "max_length", 512)
    rng = np.random.default_rng(seed)

    # ---------- 第一遍：流式写 temp + 记录每行 offset（不把全量载入内存）----------
    logger.info("流式写 temp: pos_dir=%s, neg_dir=%s, neg_ratio=%s", pos_dir, neg_dir, neg_ratio)
    offsets_list = [0]
    n_pos = 0
    with open(temp_path, "wb") as f:
        for input_text, target_text in tqdm(
            stream_pos_rows(pos_dir), desc="pos 流式写 temp", unit="row", unit_scale=True, smoothing=0.1
        ):
            line = json.dumps([input_text, target_text], ensure_ascii=False) + "\n"
            b = line.encode("utf-8")
            f.write(b)
            offsets_list.append(offsets_list[-1] + len(b))
            n_pos += 1
    if n_pos == 0:
        raise ValueError("正例目录下未读到任何数据")
    n_neg_sample = max(0, int(round(n_pos * neg_ratio)))
    with open(temp_path, "ab") as f:
        # total=None：reservoir 实际条数可能小于 n_neg_sample（负例不足时），不设 total 避免进度条停在非 100%
        for input_text, target_text in tqdm(
            reservoir_sample_neg_rows(neg_dir, n_neg_sample, rng),
            total=None,
            desc="neg 采样写 temp",
            unit="row",
            unit_scale=True,
            mininterval=0.5,
        ):
            line = json.dumps([input_text, target_text], ensure_ascii=False) + "\n"
            b = line.encode("utf-8")
            f.write(b)
            offsets_list.append(offsets_list[-1] + len(b))
    N = len(offsets_list) - 1
    offsets_arr = np.array(offsets_list, dtype=np.int64)
    np.save(offsets_path, offsets_arr, allow_pickle=False)
    del offsets_list
    logger.info("temp 总行数 N=%d (正例 %d + 负例采样 %d)", N, n_pos, N - n_pos)

    # 打乱索引（与 merge_npy_chunk 一致，只保留 inv_perm）
    perm = np.arange(N, dtype=np.int64)
    rng.shuffle(perm)
    inv_perm = np.empty(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N)
    del perm
    N_dev = int(N * dev_ratio)
    N_train = N - N_dev
    num_dev_shards = (N_dev + lines_per_shard - 1) // lines_per_shard if N_dev > 0 else 0
    num_train_shards = (N_train + lines_per_shard - 1) // lines_per_shard if N_train > 0 else 0

    # ---------- 按 chunk 划分：dev_0..dev_K + train_0..train_M（与训练集一样按 lines_per_shard 分片）----------
    chunk_data = []
    out_paths = []

    for i in range(num_dev_shards):
        start = i * lines_per_shard
        end = min((i + 1) * lines_per_shard, N_dev)
        dev_list = [(g, int(inv_perm[g]) - start) for g in range(N) if start <= inv_perm[g] < end]
        dev_list.sort(key=lambda x: x[1])
        chunk_data.append(dev_list)
        out_paths.append(os.path.abspath(os.path.join(out_dir, f"dev_{i:05d}{ext}")))

    for i in range(num_train_shards):
        start = N_dev + i * lines_per_shard
        end = min(N_dev + (i + 1) * lines_per_shard, N)
        train_list = [(g, int(inv_perm[g]) - start) for g in range(N) if start <= inv_perm[g] < end]
        train_list.sort(key=lambda x: x[1])
        chunk_data.append(train_list)
        out_paths.append(os.path.abspath(os.path.join(out_dir, f"train_{i:05d}{ext}")))

    num_chunks = len(chunk_data)
    logger.info("chunks: %d dev + %d train shards, workers=%d, format=%s", num_dev_shards, num_train_shards, workers, args.format)

    # ---------- 多进程：每个 chunk 一个任务，读 temp 按 offset tokenize 写出 ----------
    def make_task(c):
        return (
            c,
            chunk_data[c],
            out_paths[c],
            temp_path,
            offsets_path,
            model_name_or_path,
            max_seq_length,
            max_length,
            use_binary,
        )

    if workers <= 1:
        tasks = [make_task(c) for c in range(num_chunks)]
        results = []
        for t in tqdm(tasks, desc="tokenize 写出", unit="chunk"):
            results.append(_tokenize_one_chunk(t))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [make_task(c) for c in range(num_chunks)]
        results = [None] * num_chunks
        with ProcessPoolExecutor(max_workers=min(workers, num_chunks)) as executor:
            future_to_c = {executor.submit(_tokenize_one_chunk, t): t[0] for t in tasks}
            with tqdm(
                total=num_chunks,
                desc="tokenize 写出 (并行)",
                unit="chunk",
                mininterval=0.5,
            ) as pbar:
                for future in as_completed(future_to_c):
                    c = future_to_c[future]
                    results[c] = future.result()
                    pbar.update(1)
    del inv_perm, chunk_data

    # ---------- 写 filelist、删 temp ----------
    dev_filelist_lines = [f"{results[c][0]}\t{results[c][1]}" for c in range(num_dev_shards)]
    train_filelist_lines = [f"{results[c][0]}\t{results[c][1]}" for c in range(num_dev_shards, num_chunks)]
    dev_count = sum(results[c][1] for c in range(num_dev_shards))
    train_filelist_path = os.path.join(out_dir, "train.jsonl.filelist")
    with open(train_filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_filelist_lines) + "\n")
    dev_filelist_path = os.path.join(out_dir, "dev.jsonl.filelist")
    with open(dev_filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dev_filelist_lines) + "\n")
    try:
        os.remove(temp_path)
        os.remove(offsets_path)
    except OSError:
        pass
    logger.info("训练集 filelist: %s (%d 个分片)", train_filelist_path, len(train_filelist_lines))
    logger.info("验证集 filelist: %s (样本数 %d)", dev_filelist_path, dev_count)
    logger.info("完成。训练用: --train_filelist %s --eval_filelist %s", train_filelist_path, dev_filelist_path)


if __name__ == "__main__":
    main()
