# -*- coding: utf-8 -*-
"""
Qwen3 纠错对外接口：加载模型并对外提供 correct_batch / correct（与 demo、infer 一致）
"""
from typing import List, Optional

from utils.logger import logger
from qwen_model import QwenCorrectionModel


class QwenCorrector:
    """Qwen3 ASR 文本纠错器：支持从 output_dir 加载 LoRA/全量权重并预测。"""

    def __init__(
        self,
        model_name_or_path: str,
        peft_name: Optional[str] = None,
        args: Optional[dict] = None,
        use_cuda: bool = True,
        **kwargs,
    ):
        """
        :param model_name_or_path: 基座模型名或路径，如 Qwen/Qwen3-4B 或 twnlp/ChineseErrorCorrector3-4B
        :param peft_name: 训练得到的目录（含 adapter_model.safetensors 或 pytorch_model.bin），若为 None 则仅用基座
        :param args: 覆盖的参数字典，如 eval_batch_size, max_length
        """
        model_args = {"eval_batch_size": 8, "max_length": 256}
        if peft_name:
            model_args["use_peft"] = True
        if args:
            model_args.update(args)
        self.model = QwenCorrectionModel(
            model_name_or_path,
            args=model_args,
            use_cuda=use_cuda,
            **kwargs,
        )
        if peft_name:
            from peft import PeftModel
            self.model.model = PeftModel.from_pretrained(
                self.model.model, peft_name, torch_dtype=self.model.torch_dtype
            )
            self.model.model = self.model.model.merge_and_unload()
            logger.info("Loaded PEFT from %s", peft_name)
        self._system_prompt = None
        self._prefix_prompt = None

    def correct_batch(
        self,
        sentences: List[str],
        max_length: int = 256,
        batch_size: int = 16,
        prefix_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """对句子列表做纠错，返回纠正后的字符串列表。"""
        return self.model.predict(
            sentences,
            max_length=max_length,
            eval_batch_size=batch_size,
            prefix_prompt=prefix_prompt or self._prefix_prompt,
            system_prompt=system_prompt or self._system_prompt,
            **kwargs,
        )

    def correct(self, sentence: str, **kwargs) -> str:
        """单条纠错。"""
        return self.correct_batch([sentence], **kwargs)[0]
