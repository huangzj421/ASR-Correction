# ASR-Correction（Qwen3 文本纠错）

基于 **Qwen3** 的 ASR 后处理文本纠错模型，训练与推理逻辑改编自 [pycorrector](https://github.com/shibing624/pycorrector) 的 Qwen/GPT 方案。

## 特性

- 使用 **Qwen3**（如 `Qwen/Qwen3-4B` 或 `twnlp/ChineseErrorCorrector3-4B`）做生成式纠错
- 支持 **LoRA** 微调，显存友好

## 环境

```bash
pip install -r requirements.txt
# 建议：pip install transformers>=4.33 peft accelerate
```

## 使用

### 1. 预处理

与原先一致，生成按字/词分段的 `train.txt`、`dev.txt`：

```bash
python preprocess.py
```

### 2. 训练

```bash
python train.py
```

### 3. 推理 / 评测

```bash
python infer.py    # 使用 config 中的 model_dir，对 data/mandarin-accented/test.csv 评测并写 output_manacc.txt
python predict.py  # 交互式输入句子得到纠错结果
```

### 4. Demo 一键训练 + 预测

```bash
python demo.py --do_train --do_predict --dataset sighan --model_name_or_path Qwen/Qwen3-4B --model_dir output/qwen3_demo/
```

示例输出：

```
input  : 老是较书。
predict: 老是教书。

input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
predict: 感谢等五分钟以后，碰到一位很棒的女生跟我可聊。

input  : 遇到一位很棒的奴生跟我聊天。
predict: 遇到一位很棒的女生跟我聊天。

input  : 遇到一位很美的女生跟我疗天。
predict: 遇到一位很美的女生跟我聊天。

input  : 他们只能有两个选择：接受降新或自动离职。
predict: 他们只能有两个选择：接受降薪或自动离职。

input  : 王天华开心得一直说话。
predict: 王天华开心地一直说话。
```

## 说明

- 若显存不足，可在 `config` 或 `model_args` 中开启 `int4=True` / `qlora=True`（需安装 `bitsandbytes`）。