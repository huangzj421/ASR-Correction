# ASR-Correction

基于 **Qwen3** 的 ASR 后处理文本纠错项目，支持通用纠错与**上下文关键词增强纠错**两种模式，训练与推理逻辑参考 [pycorrector](https://github.com/shibing624/pycorrector) 的 Qwen/GPT 方案。

---

## 特性

- **生成式纠错**：使用 Qwen3（如 `Qwen/Qwen3-4B`）或 [ChineseErrorCorrector3-4B](https://huggingface.co/twnlp/ChineseErrorCorrector3-4B) 做序列到序列纠错
- **LoRA 微调**：显存友好，支持 PEFT 适配
- **关键词纠错**：输入「错误文本 + 候选关键词」，模型在关键词约束下纠错，适合 ASR 置信词/关键词场景
- **CEC3 基线**：提供 `run_infer_cec3.py`，用 CEC3 模型对同一输入做纠错对比（不利用关键词）

---

## 环境

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`transformers`、`peft`、`accelerate`、`datasets`、`jieba`、`jiwer`、`sacrebleu` 等。若显存不足，可在配置中开启 `int4=True` / `qlora=True`（需安装 `bitsandbytes`）。

---

## 项目结构概览

| 路径 | 说明 |
|------|------|
| `config.py` | 主配置：数据路径、基座模型、关键词正/负例目录、训练/推理参数 |
| `config_corrector3.py` | CEC3 相关配置（如用 CEC3 做基座或对比实验） |
| `preprocess.py` | 通用纠错数据预处理，生成 `train_*.txt` / `dev_*.txt`（按字/词分段） |
| `train.py` | 通用纠错训练入口 |
| `train_keyword.py` | **关键词纠错**训练：正例目录 + 负例目录，格式 `错误<mask_99>关键词<mask_99>正确` |
| `infer.py` | 推理入口：加载 PEFT/全量模型，支持 `config.model_dir` 或指定 checkpoint |
| `predict.py` | 交互式纠错：输入句子，输出纠错结果 |
| `run_infer.py` | 批量关键词推理：输入每行 `序号 错误文本<mask_99>关键词<mask_99>`，对多个 checkpoint 输出 `序号 纠错文本` |
| `run_infer_cec3.py` | 使用 CEC3 对相同格式输入纠错（仅用错误文本，不用关键词），输出 `序号 纠错文本` |
| `demo.py` | 一键预处理 + 训练 + 预测示例（SIGHAN 等） |
| `data_reader.py` | 数据读取：通用 `load_bert_data`、关键词 `load_keyword_correction_data`、`build_keyword_input` |
| `qwen_model.py` / `qwen_utils.py` | Qwen3 模型封装与预处理工具 |
| `scripts/prepare_keyword_correction_jsonl.py` | 流式 + 多进程生成关键词纠错 jsonl/bin 数据（大表用） |

---

## 数据格式

### 通用纠错（SIGHAN / Mandarin 等）

- **原始**：TSV/CSV，如 `source\target` 或 `recognition,reference`
- **预处理后**：`train_*.txt` / `dev_*.txt`，每行 `input_text\ttarget_text`（可按字/词分段，由 `config.use_segment`、`segment_type` 控制）

### 关键词纠错

- **训练/推理统一格式**：`错误文本<mask_99>关键词1, 关键词2, ...<mask_99>正确文本`（训练时）；推理输入见下。
- **训练数据目录**：`config.keyword_pos_dir`（正例）、`config.keyword_neg_dir`（负例），其下若干 `.txt`，每行一条上述格式；负例按 `keyword_neg_ratio` 采样后与正例混合。
- **推理输入**（`run_infer.py`）：每行  
  `序号 错误文本<mask_99>关键词<mask_99>`  
  脚本会解析并构造成模型输入（错误文本 + 候选关键词），输出每行 `序号 纠错后的文本`。

---

## 使用说明

### 1. 通用纠错：预处理 → 训练 → 推理

```bash
# 预处理（生成 train_*.txt / dev_*.txt）
python preprocess.py

# 训练（依赖 config：train_path, dev_path, model_name_or_path, model_dir 等）
python train.py

# 推理：对 data/mandarin-accented/test.csv 评测并写 output_manacc.txt
python infer.py

# 交互式预测
python predict.py
```

### 2. 关键词纠错：训练 → 批量推理

在 `config.py` 中配置 `keyword_pos_dir`、`keyword_neg_dir`、`keyword_model_dir` 等后：

```bash
# 训练
python train_keyword.py

# 批量推理：对每个 checkpoint 输出 pred_<checkpoint>.txt
python run_infer.py \
  --input_txt /path/to/prompt.txt \
  --model_dir output/model_qwen3_keyword \
  --output_dir output/model_qwen3_keyword/infer_output \
  --max_length 1024 \
  --batch_size 1
```

`input_txt` 每行格式：`序号 错误文本<mask_99>关键词<mask_99>`。

### 3. 使用 CEC3 做纠错（基线 / 对比）

输入格式与 `run_infer.py` 相同（每行 `序号 错误文本<mask_99>关键词<mask_99>`），脚本只使用「错误文本」调用 CEC3，输出每行 `序号 纠错后的文本`：

```bash
# 使用默认 CEC3 模型与 output 路径
bash run_infer_cec3.sh prompt.txt

# 或直接调用
python run_infer_cec3.py \
  --input_txt prompt.txt \
  --output_txt output/corrector3_ori.txt \
  --max_new_tokens 512 \
  --batch_size 8
```

指定本地模型：`--model_name /path/to/ChineseErrorCorrector3-4B`。

### 4. Demo：一键训练 + 预测

```bash
python demo.py --do_train --do_predict \
  --dataset sighan \
  --model_name_or_path Qwen/Qwen3-4B \
  --model_dir output/qwen3_demo/
```

示例输出：

```
input  : 老是较书。
predict: 老是教书。

input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
predict: 感谢等五分钟以后，碰到一位很棒的女生跟我可聊。
```

---

## 配置说明

- **基座模型**：`config.model_name_or_path` 为 HuggingFace 模型 id 或本地路径；离线环境请填本地绝对路径。
- **通用纠错**：`dataset`（sighan / manacc 等）、`train_path` / `dev_path`、`model_dir`、`batch_size`、`epochs`、`max_length`、`use_peft`、`learning_rate`、`eval_steps`、`save_steps`。
- **关键词纠错**：`keyword_pos_dir`、`keyword_neg_dir`、`keyword_neg_ratio`、`keyword_dev_ratio`、`keyword_model_dir`；`train_keyword.py` 会从正/负例目录读入并划分验证集。
- **显存**：可开启 `int4=True` / `qlora=True`（需 `bitsandbytes`）。

---

## 其他

- **WER 评测**：`run_wer.sh` 中需配置本机 `WER` 可执行文件、参考文本 `REF` 和推理结果目录，对 `pred_*.txt` 等计算 WER。
- **大数据预处理**：使用 `scripts/prepare_keyword_correction_jsonl.py` 做流式 + 多进程生成 jsonl/bin，用法见脚本内注释。
