# BERTSeq2Seq Model


## Features

* 基于BERT的Sequence to Sequence模型


## Usage

### Requirements
* pip安装依赖包

```
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
```

## Demo

- bertseq2seq demo

示例[demo.py](./demo.py)
```
python demo.py --do_train --do_predict
```

## Detail

### Preprocess

```
python preprocess.py

generate toy train data(`train.txt`) and valid data(`dev.txt`), segment by char.
```

```
* train.txt:

如 服 装 ， 若 有 一 个 很 流 行 的 形 式 ， 人 们 就 赶 快 地 追 求 。\t如 服 装 ， 若 有 一 个 很 流 行 的 样 式 ， 人 们 就 赶 快 地 追 求 。
```

### Train

```
python train.py
```

### Infer

```
python infer.py

```

### Result

```
input  : 老是较书。
predict: 老是教书

input  : 感谢等五分以后，碰到一位很棒的奴生跟我可聊。
predict: 感谢等五分以后又碰到一位很棒的男生跟我可聊

input  : 遇到一位很棒的奴生跟我聊天。
predict: 遇到一位很棒的男生跟我聊天

input  : 遇到一位很美的女生跟我疗天。
predict: 遇到一位很美的女生跟我聊天

input  : 他们只能有两个选择：接受降新或自动离职。
predict: 他们只能有两个选择先接受降薪或自动离职

input  : 王天华开心得一直说话。
predict: 王天华开心地一直说话

```