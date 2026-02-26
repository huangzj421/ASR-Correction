# -*- coding: utf-8 -*-
"""Qwen3 ASR 纠错配置（已移除 BERTSeq2Seq）"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

sighan_train_path = os.path.join(pwd_path, "./data/sighan_2015/train.tsv")
sighan_dev_path = os.path.join(pwd_path, "./data/sighan_2015/dev.tsv")
speechx_train_path = os.path.join(pwd_path, "./data/mandarin-accented/train.csv")
speechx_dev_path = os.path.join(pwd_path, "./data/mandarin-accented/valid.csv")

use_segment = True
segment_type = "char"

dataset = "sighan"  # manacc / sighan / aishell1 / magicdata / mandarin
output_dir = os.path.join(pwd_path, "output")
train_path = os.path.join(output_dir, "train_{}.txt".format(dataset))
dev_path = os.path.join(output_dir, "dev_{}.txt".format(dataset))

# Qwen3 基座：HuggingFace 模型 id 或本地目录（离线时填绝对路径，如 /data/models/Qwen3-4B）
model_name_or_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/hf_models/huggingface.co/twnlp/ChineseErrorCorrector3-4B"
model_dir = os.path.join(output_dir, "model_qwen3_{}".format(dataset))
# model_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/ASR-Correction/output/model_qwen3_keyword/checkpoint-14400"

batch_size = 4
epochs = 3
max_length = 512
max_seq_length = 512
eval_batch_size = 16
eval_steps = 200
save_steps = 400
use_peft = True
learning_rate = 2e-5

gpu_id = 0

# ---------- 上下文关键词纠错分支（train_keyword.py）----------
# 正例目录：正确词在关键词里；负例目录：全干扰，希望模型不纠错
keyword_pos_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_mix_no_homo/raw_data_test"
keyword_neg_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_distract/raw_data_test"
# 负例采样量 = 正例数量 * keyword_neg_ratio
keyword_neg_ratio = 1.0
# 从合并后的数据中划分验证集比例
keyword_dev_ratio = 0.1
keyword_model_dir = os.path.join(output_dir, "model_qwen3_keyword_corrector3")
manual_seed = 42

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
