# -*- coding: utf-8 -*-
"""
Qwen3 文本纠错训练工具：数据集与参数（摘编自 pycorrector.gpt.gpt_utils，适配 ASR 纠错 input_text/target_text 格式）
"""
import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
from typing import List, Optional, Dict, Sequence

from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from utils.logger import logger

IGNORE_INDEX = LabelSmoother.ignore_index


@dataclass
class QwenArgs:
    """Qwen3 纠错模型训练参数（对应 pycorrector GptArgs）"""
    model_class: str = "QwenArgs"
    dataset_class: Dataset = None
    learning_rate: float = 2e-5
    manual_seed: int = 42
    fp16: bool = False
    bf16: bool = False
    int8: bool = False
    int4: bool = False
    debug: bool = False
    max_seq_length: int = 256
    max_length: int = 256
    warmup_steps: int = 50
    report_to: str = "tensorboard"
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 400
    max_eval_samples: int = 20
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    do_sample: bool = True
    temperature: float = 0.1
    special_tokens_list: list = field(default_factory=list)
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    model_name: str = None
    tokenizer_name: str = None
    reprocess_input_data: bool = False
    silent: bool = False
    no_cache: bool = False
    cache_dir: str = "cache_dir/"
    no_save: bool = False
    top_k: float = 40
    top_p: float = 0.9
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-4B")
    use_peft: bool = True
    peft_type: str = "LORA"
    peft_bin_name: str = "adapter_model.bin"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["all"])
    lora_bias: str = "none"
    adalora_init_r: int = 12
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    lora_beta: float = 0.85
    num_virtual_tokens: int = 20
    prompt_encoder_hidden_size: int = 128
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    save_total_limit: int = 10
    remove_unused_columns: bool = False
    logging_steps: int = 50
    resume_from_checkpoint: str = None
    torch_compile: bool = False
    trust_remote_code: bool = True
    qlora: bool = False
    preprocessing_num_workers: int = 4
    prompt_template_name: str = "qwen"

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            raise TypeError(f"{new_values} is not a Python dict.")


# Qwen 对话模板：与 pycorrector qwen 一致，用于构造 input_ids/labels
SYSTEM_PROMPT = "你是一个中文文本纠错助手，专门用于ASR识别结果的后处理。请根据用户提供的识别文本，只输出纠正后的句子，不要解释。"
USER_PROMPT_PREFIX = "对以下ASR识别结果进行文本纠错，只输出纠正后的句子。\n\n"


def build_qwen_prompt_and_response(input_text: str, target_text: str) -> List[str]:
    """构造单条 (system+user, assistant) 对话内容，返回 [user_part, assistant_part]。"""
    user_part = USER_PROMPT_PREFIX + input_text.strip()
    assistant_part = target_text.strip()
    return [user_part, assistant_part]


def preprocess_one_correction_pair(
    input_text: str,
    target_text: str,
    tokenizer,
    args: QwenArgs,
) -> Dict[str, List[int]]:
    """
    单条 (input_text, target_text) 转为 input_ids 和 labels（list of int）。
    供预处理写 jsonl 与流式 Dataset 复用。
    """
    user_part, assistant_part = build_qwen_prompt_and_response(input_text, target_text)
    max_full_length = args.max_seq_length + args.max_length

    messages_prompt_only = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_part},
        {"role": "assistant", "content": " "},
    ]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt_only,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            truncation=True,
            max_length=args.max_seq_length,
            enable_thinking=False,
        )
        if isinstance(prompt_ids, list) and len(prompt_ids):
            if isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            elif not isinstance(prompt_ids[0], int):
                prompt_ids = list(prompt_ids[0])
        prompt_ids = prompt_ids[:-1]
    except Exception:
        prompt_ids = _encode_prompt_only(tokenizer, user_part, args.max_seq_length)
    prompt_len = len(prompt_ids)

    messages_full = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_part},
        {"role": "assistant", "content": assistant_part},
    ]
    try:
        full_ids = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            truncation=True,
            max_length=max_full_length,
            enable_thinking=False,
        )
        if isinstance(full_ids, list) and len(full_ids) and isinstance(full_ids[0], (list, int)):
            if isinstance(full_ids[0], list):
                full_ids = full_ids[0]
    except Exception:
        full_ids = _encode_qwen_style(
            tokenizer, SYSTEM_PROMPT, user_part, assistant_part, max_full_length
        )
    input_ids = full_ids[:max_full_length]
    labels = [IGNORE_INDEX] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]
    return dict(input_ids=input_ids, labels=labels)


def preprocess_correction_pairs(
    input_texts: List[str],
    target_texts: List[str],
    tokenizer,
    args: QwenArgs,
) -> Dict[str, List]:
    """
    将 (input_text, target_text) 对转为 causal LM 的 input_ids 和 labels。
    只对 assistant 回复部分计算 loss，前面全部用 IGNORE_INDEX。
    """
    input_ids_list = []
    labels_list = []
    for input_text, target_text in zip(input_texts, target_texts):
        one = preprocess_one_correction_pair(input_text, target_text, tokenizer, args)
        input_ids_list.append(one["input_ids"])
        labels_list.append(one["labels"])
    return dict(input_ids=input_ids_list, labels=labels_list)


def _encode_prompt_only(tokenizer, user_content: str, max_len: int) -> List[int]:
    """仅编码 system + user + assistant 开头（无回复内容）。"""
    parts = [
        "<|im_start|>system\n", SYSTEM_PROMPT, "<|im_end|>\n",
        "<|im_start|>user\n", user_content, "<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
    text = "".join(parts)
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)


def _encode_qwen_style(
    tokenizer, system_prompt: str, user_content: str, assistant_content: str, max_len: int
) -> List[int]:
    """简易 Qwen 风格编码：system + user + assistant。"""
    parts = [
        "<|im_start|>system\n", system_prompt, "<|im_end|>\n",
        "<|im_start|>user\n", user_content, "<|im_end|>\n",
        "<|im_start|>assistant\n", assistant_content, "<|im_end|>",
    ]
    text = "".join(parts)
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)


def filter_empty_labels(example):
    """过滤掉 labels 全为 IGNORE_INDEX 的样本。"""
    return not all(label == IGNORE_INDEX for label in example["labels"])


class QwenCorrectionDataset(Dataset):
    """从 (input_text, target_text) DataFrame 或 list 构建的 Qwen 纠错数据集。"""

    def __init__(self, tokenizer, args: QwenArgs, data, mode: str = "train"):
        self.tokenizer = tokenizer
        self.args = args
        cached_features_file = os.path.join(
            args.cache_dir,
            (args.model_name or "qwen").replace("/", "_")
            + "_asr_correction_cached_"
            + str(args.max_seq_length)
            + "_"
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and not args.no_cache)
        ):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as f:
                self.examples = pickle.load(f)
        else:
            os.makedirs(args.cache_dir, exist_ok=True)
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame) and "input_text" in data.columns:
                    input_texts = data["input_text"].astype(str).tolist()
                    target_texts = data["target_text"].astype(str).tolist()
                else:
                    raise AttributeError
            except Exception:
                input_texts = [x[0] for x in data]
                target_texts = [x[1] for x in data]
            if mode == "dev" and args.max_eval_samples is not None:
                n = min(len(input_texts), args.max_eval_samples)
                input_texts = input_texts[:n]
                target_texts = target_texts[:n]
            out = preprocess_correction_pairs(input_texts, target_texts, tokenizer, args)
            self.examples = [{"input_ids": o, "labels": l} for o, l in zip(out["input_ids"], out["labels"])]
            if not args.no_cache:
                with open(cached_features_file, "wb") as f:
                    pickle.dump(self.examples, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _read_one_sample_binary(f, dtype=np.int32):
    """从已打开的二进制文件读一条样本，返回 {"input_ids": list, "labels": list}。"""
    n_i = np.frombuffer(f.read(4), dtype=np.int32)[0]
    input_ids = np.frombuffer(f.read(int(n_i) * 4), dtype=dtype).tolist()
    n_l = np.frombuffer(f.read(4), dtype=np.int32)[0]
    labels = np.frombuffer(f.read(int(n_l) * 4), dtype=dtype).tolist()
    return {"input_ids": input_ids, "labels": labels}


class QwenCorrectionIndexDataset(Dataset):
    """
    按 filelist 流式加载：只把「文件路径 + 每文件行数」放在内存，
    __getitem__(idx) 时再打开对应文件读该条，返回 {"input_ids", "labels"}。
    支持 jsonl（每行一个 JSON）或 binary（--format binary 生成 .bin + .bin.offsets.npy）。
    """

    def __init__(self, filelist_path: str):
        self.data_files = []
        self.idata = [0]
        self._bin_offsets = {}  # file_id -> np.array(offsets)，binary 时按需加载
        count = 0
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
                fn = parts[0].rstrip(":")
                c = int(parts[1]) if len(parts) > 1 else 0
                if c <= 0:
                    continue
                count += c
                self.data_files.append(fn)
                self.idata.append(count)
        assert len(self.data_files) > 0, f"filelist 为空或格式错误: {filelist_path}"
        self.total_sample_len = count

    def __len__(self):
        return self.total_sample_len

    def _get_bin_offsets(self, file_id: int):
        if file_id not in self._bin_offsets:
            path = self.data_files[file_id]
            off_path = path + ".offsets.npy"
            self._bin_offsets[file_id] = np.load(off_path, allow_pickle=False)
        return self._bin_offsets[file_id]

    def __getitem__(self, idx: int):
        import bisect
        import linecache
        file_id = bisect.bisect_right(self.idata, idx) - 1
        line_id = idx - self.idata[file_id]
        path = self.data_files[file_id]
        if path.endswith(".bin"):
            offsets = self._get_bin_offsets(file_id)
            start = int(offsets[line_id])
            end = int(offsets[line_id + 1])
            with open(path, "rb") as f:
                f.seek(start)
                raw = f.read(end - start)
            import io
            return _read_one_sample_binary(io.BytesIO(raw))
        line = linecache.getline(path, line_id + 1)
        if not line:
            raise IndexError(f"idx={idx} file_id={file_id} line_id={line_id} path={path}")
        return json.loads(line.strip())
