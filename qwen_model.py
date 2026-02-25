# -*- coding: utf-8 -*-
"""
Qwen3 ASR 文本纠错模型：训练与推理（摘编自 pycorrector.gpt.gpt_model，适配本项目）
"""
import inspect
import math
import os
import random
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME

from utils.logger import logger
from qwen_utils import QwenArgs, QwenCorrectionDataset, IGNORE_INDEX

has_cuda = torch.cuda.is_available()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _extract_corrected_sentence(raw: str) -> str:
    """从模型输出中取出仅纠错结果：去掉<think>块，取</think>后的首句或首行。"""
    import re
    text = raw.strip()
    if not text:
        return ""
    # 若有 </think>，取该标签之后的内容作为候选（模型常把答案放在</think>后）
    if "</think>" in text:
        idx = text.find("</think>")
        text = text[idx + 8 :].strip()  # len("</think>") == 7
    # 去掉可能残留的 <think>...</think> 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    if not text:
        return ""
    # 取第一行作为纠错结果
    first_line = text.split("\n")[0].strip()
    return first_line if first_line else text.strip()


def _use_local_files_only(model_name_or_path: str) -> bool:
    """离线加载：当为本地目录或设置了 HF_HUB_OFFLINE 时只读本地/缓存，不访问网络。"""
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
        return True
    if os.path.isdir(model_name_or_path) and os.path.isfile(os.path.join(model_name_or_path, "config.json")):
        return True
    return False


class QwenCorrectionModel:
    """基于 Qwen3 的 ASR 文本纠错模型：训练与预测。"""

    def __init__(
        self,
        model_name_or_path: str,
        args: Optional[Union[dict, QwenArgs]] = None,
        use_cuda: bool = has_cuda,
        cuda_device: int = -1,
        **kwargs,
    ):
        self.args = QwenArgs()
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, QwenArgs):
            self.args = args

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.device_map = "auto"
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda" if cuda_device == -1 else f"cuda:{cuda_device}")
            self.device_map = {"": cuda_device} if cuda_device >= 0 else "auto"
        else:
            self.device = torch.device("cpu")
            self.device_map = {"": "cpu"}
            self.args.fp16 = False
            self.args.int8 = False

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp = self.world_size != 1
        if self.ddp:
            self.device_map = {"": self.local_rank}

        self.results = {}
        if self.args.bf16:
            self.args.fp16 = False
        if self.args.fp16:
            self.args.bf16 = False
        self.torch_dtype = (
            torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        )

        local_files_only = _use_local_files_only(model_name_or_path)
        if local_files_only:
            logger.info("Offline mode: loading from local path or cache only (local_files_only=True)")

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.args.trust_remote_code,
            local_files_only=local_files_only,
            **kwargs,
        )
        _load_kw = dict(
            config=self.config,
            load_in_8bit=self.args.int8,
            load_in_4bit=self.args.int4,
            low_cpu_mem_usage=not is_deepspeed_zero3_enabled(),
            device_map=self.device_map,
            trust_remote_code=self.args.trust_remote_code,
            local_files_only=local_files_only,
            quantization_config=(
                BitsAndBytesConfig(
                    load_in_4bit=self.args.int4,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                )
                if getattr(self.args, "qlora", False)
                else None
            ),
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dtype=self.torch_dtype, **_load_kw
            )
        except TypeError:
            _load_kw["torch_dtype"] = self.torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **_load_kw)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.args.trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = " "
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
        if getattr(self.model.config, "architectures", None) and self.model.config.architectures:
            arch = self.model.config.architectures[0]
            if arch in ("Qwen2ForCausalLM", "Qwen3ForCausalLM"):
                self.tokenizer.padding_side = "left"

        self.args.model_name = model_name_or_path

    def train_model(
        self,
        train_data,
        output_dir=None,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """训练模型。train_data / eval_data 为 DataFrame（含 input_text, target_text）或 (input_texts, target_texts) list。"""
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            PeftModel,
            prepare_model_for_kbit_training,
            set_peft_model_state_dict,
        )

        if args:
            self.args.update_from_dict(args)
        if not output_dir:
            output_dir = self.args.output_dir
        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Set overwrite_output_dir=True to overwrite."
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            gradient_checkpointing=self.args.gradient_checkpointing,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            optim=self.args.optimizer,
            save_strategy=self.args.save_strategy,
            eval_strategy="steps" if eval_data is not None else "no",
            eval_steps=self.args.eval_steps if eval_data is not None else None,
            load_best_model_at_end=bool(eval_data is not None),
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            remove_unused_columns=self.args.remove_unused_columns,
            report_to=self.args.report_to,
            overwrite_output_dir=self.args.overwrite_output_dir,
            no_cuda=(self.device.type == "cpu"),
            **kwargs,
        )
        resume_from_checkpoint = self.args.resume_from_checkpoint

        if self.args.use_peft:
            if self.args.int8 or self.args.int4:
                self.model = prepare_model_for_kbit_training(
                    self.model, self.args.gradient_checkpointing
                )
            target_modules = self.args.lora_target_modules
            if isinstance(target_modules, list) and "all" in target_modules:
                target_modules = self._find_all_linear_names()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=target_modules,
                bias=self.args.lora_bias,
            )
            if isinstance(self.model, PeftModel):
                self.model = self.model.merge_and_unload()
            self.model = get_peft_model(self.model, peft_config)
            for p in filter(lambda x: x.requires_grad, self.model.parameters()):
                p.data = p.data.to(torch.float32)
            if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
                adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
                if not os.path.isfile(adapter_path):
                    adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.bin")
                if os.path.isfile(adapter_path):
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, resume_from_checkpoint)
                    logger.info("Resumed from checkpoint %s", resume_from_checkpoint)
                else:
                    resume_from_checkpoint = None
            self.model.print_trainable_parameters()
        else:
            os.makedirs(output_dir, exist_ok=True)

        train_dataset = self.load_and_cache_examples(train_data)
        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True) if eval_data is not None else None

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
        self.model.enable_input_require_grads()

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, label_pad_token_id=IGNORE_INDEX, padding=True
        )
        _trainer_kw = dict(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        # 新版本 Trainer 使用 processing_class 替代 tokenizer
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            _trainer_kw["processing_class"] = self.tokenizer
        else:
            _trainer_kw["tokenizer"] = self.tokenizer
        trainer = Trainer(**_trainer_kw)

        logger.info("*** Train ***")
        global_step, training_loss, metrics = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.results.update(metrics)
        trainer.save_state()
        self.save_model(model=self.model, output_dir=output_dir)
        self.model.config.use_cache = True

        if eval_dataset is not None:
            logger.info("*** Evaluate ***")
            if self.args.fp16:
                self.model.half()
            eval_metrics = trainer.evaluate(metric_key_prefix="eval")
            try:
                eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
            except OverflowError:
                eval_metrics["perplexity"] = float("inf")
            self.results.update(eval_metrics)

        if verbose:
            logger.info("Training complete. Saved to %s.", output_dir)
        return global_step, training_loss

    def _find_all_linear_names(self):
        import torch.nn as nn
        names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "lm_head" in name or "output_layer" in name:
                    continue
                names.add(name.split(".")[-1])
        return list(names)

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """构建训练/验证数据集。data 为 DataFrame(input_text, target_text) 或 list of [src, trg]。"""
        if data is None:
            return None
        return QwenCorrectionDataset(
            self.tokenizer, self.args, data, mode="dev" if evaluate else "train"
        )

    @torch.inference_mode()
    def predict(
        self,
        sentences: List[str],
        max_length: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        system_prompt: Optional[str] = None,
        prefix_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """对句子列表做纠错预测，返回纠正后的字符串列表。"""
        from qwen_utils import USER_PROMPT_PREFIX, SYSTEM_PROMPT

        self.model.eval()
        if self.args.fp16:
            self.model.half()
        batch_size = eval_batch_size or self.args.eval_batch_size
        max_len = max_length or self.args.max_length
        sys_prompt = system_prompt or SYSTEM_PROMPT
        prefix = prefix_prompt or USER_PROMPT_PREFIX

        all_outputs = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Predict", disable=self.args.silent):
            batch = sentences[i: i + batch_size]
            user_inputs = [prefix + s.strip() for s in batch]
            conversations = []
            for u in user_inputs:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": u},
                ]
                conversations.append(messages)
            inputs = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_length,
            )
            if hasattr(inputs, "to"):
                input_ids = inputs.to(self.device)
            else:
                input_ids = torch.tensor(inputs, device=self.device)
            prompt_length = input_ids.shape[1]
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_len,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            for j, out in enumerate(outputs):
                gen = out[prompt_length:]
                text = self.tokenizer.decode(gen, skip_special_tokens=True)
                text = _extract_corrected_sentence(text)
                all_outputs.append(text)
        return all_outputs

    def save_model(self, output_dir=None, model=None, **kwargs):
        import json
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if model is not None and not self.args.no_save:
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            with open(os.path.join(output_dir, "model_args.json"), "w", encoding="utf-8") as f:
                json.dump({"model_name_or_path": self.args.model_name}, f, ensure_ascii=False)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
