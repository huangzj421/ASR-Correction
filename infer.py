# -*- coding: utf-8 -*-
"""
Qwen3 ASR 纠错推理入口（已移除 BERTSeq2Seq，仅保留 Qwen3）
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.logger import logger
from qwen_model import QwenCorrectionModel


def load_base_model_name(model_dir):
    """从 model_dir 的 model_args.json 读取基座模型名。"""
    path = os.path.join(model_dir, "model_args.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_name_or_path")
    return getattr(config, "model_name_or_path", "Qwen/Qwen3-4B")


class Inference:
    """Qwen3 纠错推理：从 model_dir 加载基座 + PEFT 并预测。"""

    def __init__(
        self,
        model_dir,
        max_length=256,
        eval_batch_size=8,
        use_cuda=None,
        checkpoint_path=None,
    ):
        """
        :param model_dir: 根模型目录（含 model_args.json），或单个 checkpoint 目录。
        :param checkpoint_path: 若提供，则 base 名从 model_dir 读，PEFT 从 checkpoint_path 加载（用于多 checkpoint 推理）。
        """
        use_cuda = use_cuda if use_cuda is not None else torch.cuda.is_available()
        load_dir = checkpoint_path if checkpoint_path else model_dir
        base_name = load_base_model_name(model_dir)
        args = {
            "max_length": max_length,
            "eval_batch_size": eval_batch_size,
        }
        self.model = QwenCorrectionModel(base_name, args=args, use_cuda=use_cuda)
        if os.path.isdir(load_dir):
            adapter_path = os.path.join(load_dir, "adapter_config.json")
            if os.path.isfile(adapter_path):
                from peft import PeftModel
                self.model.model = PeftModel.from_pretrained(
                    self.model.model, load_dir, torch_dtype=self.model.torch_dtype
                )
                self.model.model = self.model.model.merge_and_unload()
                logger.info("Loaded PEFT from %s", load_dir)
            else:
                self.model.model = AutoModelForCausalLM.from_pretrained(
                    load_dir,
                    torch_dtype=self.model.torch_dtype,
                    device_map=self.model.device_map,
                    local_files_only=True,
                )
                self.model.tokenizer = AutoTokenizer.from_pretrained(load_dir, local_files_only=True)
                logger.info("Loaded full model from %s", load_dir)
        self.max_length = max_length

    def predict(self, sentence_list):
        return self.model.predict(sentence_list, max_length=self.max_length)


if __name__ == "__main__":
    from jiwer import wer
    import sacrebleu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(
        config.model_dir,
        max_length=config.max_length,
        eval_batch_size=getattr(config, "eval_batch_size", 16),
    )
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(pwd_path, "./data/mandarin-accented/test.csv")
    save_path = os.path.join(pwd_path, "output_manacc.txt")
    source_list = []
    target_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("recognition"):
                continue
            parts = line.split(",")
            source_list.append(parts[0])
            target_list.append(parts[1])

    corrected_list = inf.predict(source_list)
    BLEU_src = sacrebleu.corpus_bleu(source_list, [target_list], tokenize="zh").score
    BLEU_corr = sacrebleu.corpus_bleu(corrected_list, [target_list], tokenize="zh").score
    count = len(source_list)
    CER_src = sum(wer(" ".join(list(t.strip())), " ".join(list(s.strip()))) for s, t in zip(source_list, target_list)) / count
    CER_corr = sum(wer(" ".join(list(t.strip())), " ".join(list(c.strip()))) for c, t in zip(corrected_list, target_list)) / count

    with open(save_path, "a", encoding="utf-8") as f:
        for src, corr, tgt in zip(source_list, corrected_list, target_list):
            f.write("\nsrc:%s\ncorr:%s\ntgt:%s\n" % (src, corr, tgt))
        f.write("\nBLEU_src:%s => BLEU_corr:%s\n" % (BLEU_src, BLEU_corr))
        f.write("CER_src:%s => CER_corr:%s\n" % (CER_src, CER_corr))
