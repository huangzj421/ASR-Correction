# ASR-Correction（Qwen3 文本纠错）

基于 **Qwen3** 的 ASR 后处理文本纠错模型，训练与推理逻辑摘编自 [pycorrector](https://github.com/shibing624/pycorrector) 的 Qwen/GPT 方案，**已移除 BERTSeq2Seq**，仅使用 Qwen3（支持 LoRA 微调）。

## 特性

- 使用 **Qwen3**（如 `Qwen/Qwen3-4B` 或 `twnlp/ChineseErrorCorrector3-4B`）做生成式纠错
- 支持 **LoRA** 微调，显存友好
- 数据格式与原有项目一致：`input_text \t target_text`（空格分词或不分词）

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

或通过 demo 指定数据与超参：

```bash
python demo.py --do_train --dataset sighan --model_name_or_path Qwen/Qwen3-4B --model_dir output/qwen3_demo/
```

### 3. 推理 / 评测

```bash
python infer.py    # 使用 config 中的 model_dir，对 data/mandarin-accented/test.csv 评测并写 output_manacc.txt
python predict.py  # 交互式输入句子得到纠错结果
```

### 4. Demo 一键训练 + 预测

```bash
python demo.py --do_train --do_predict
```

示例输出：

```
input  : 老是较书。
predict: 老是教书

input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
predict: 感谢等五分以后又碰到一位很棒的男生跟我可聊
```

## 离线下载并加载 Qwen3-4B

`model_name_or_path` 支持 **HuggingFace 模型 id** 或 **本地目录绝对路径**。离线使用步骤：

### 1. 在有网络的机器上下载到本地目录

```bash
# 安装 huggingface_hub（若未安装）
pip install huggingface_hub

# 下载到指定目录（会得到 snapshot 下的模型文件）
huggingface-cli download Qwen/Qwen3-4B --local-dir /path/to/Qwen3-4B

# 或使用 Python 拉取
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-4B', local_dir='/path/to/Qwen3-4B')
"
```

将 `/path/to/Qwen3-4B` 拷到离线机（或内网共享路径）。

### 2. 配置为本地路径

在 **config.py** 中改为本地目录：

```python
# 改为你本机的 Qwen3-4B 目录（绝对路径更稳妥）
model_name_or_path = "/path/to/Qwen3-4B"
```

或命令行传参（demo）：

```bash
python demo.py --do_train --model_name_or_path /path/to/Qwen3-4B --model_dir output/qwen3_demo/
```

训练和推理都会从该目录读权重，**无需联网**。若使用 ModelScope，可先 `modelscope download` 到本地，同样把 `model_name_or_path` 指到该目录即可。

---

## 配置说明

- `config.py`：`model_name_or_path`（基座：HF id 或**本地路径**）、`model_dir`（保存/加载目录）、`batch_size`、`epochs`、`max_length`、`use_peft` 等
- 训练数据：`train_*.txt` / `dev_*.txt`，每行：`输入句子\t目标句子`（可空格分段）

## 与 pycorrector 的对应关系

- **训练**：对应 pycorrector 的 `GptModel` + `GptSupervisedDataset`，此处用 `QwenCorrectionModel` + `QwenCorrectionDataset`，数据格式为 `(input_text, target_text)`，内部转为 Qwen chat 模板并只对 assistant 部分算 loss。
- **推理**：与 pycorrector 的 `GptCorrector` / `GptModel.predict` 一致，支持从 `model_dir` 加载基座 + PEFT 后做纠错。

## 说明

- 旧版 BERTSeq2Seq 相关代码（`seq2seq_model.py`、`seq2seq_utils.py`、`model_args.py`）已不再被本流程使用，可保留或删除。
- 若显存不足，可在 `config` 或 `model_args` 中开启 `int4=True` / `qlora=True`（需安装 `bitsandbytes`）。
