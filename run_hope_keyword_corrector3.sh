#!/usr/bin/bash
# Hope 多节点多卡提交脚本：训练 train_keyword_corrector3.py（模仿 embr_hongyu/run_1d.sh）

# 与 run_1d.sh 一致：hope_submit_gpu.py 所在仓库根路径，请按本机环境修改
export SPEECH_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/xianghongyu/code/speech
export AM_ROOT=$SPEECH_ROOT/asr/offline/am_training

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

umask 000

# # Hope 任务会 source .hope/path.sh，用于设置运行时环境
# mkdir -p .hope
# touch .hope/path.sh
# chmod +x .hope/path.sh
# echo "export SPEECH_ROOT=$SPEECH_ROOT" > .hope/path.sh
# echo "export AM_ROOT=\$SPEECH_ROOT/asr/offline/am_training" >> .hope/path.sh
# echo "export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\"" >> .hope/path.sh

stage=2

### hope ###
use_hope=true
job_name=a0302_p0_64x30_4_asr_correction_keyword_corrector3
nnodes=4
nproc_per_node=8
gpu_type=gcoresh800-80g
image=registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/hdp_training_cuda12.1.1_python39_mllm_moe_rm_te_1.0.0_167501f0
gpu_queue=root.hldy_training_cluster.hadoop-aipnlp.a_exp

nprocs=$((nnodes * nproc_per_node))
dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/ASR-Correction/output/model_qwen3_keyword_corrector3_dense2b_text_itn"
hope_log_dir=$dir/hope_log
mkdir -p $hope_log_dir

# 大数据流式：先跑 scripts/prepare_keyword_correction_jsonl.py 得到 filelist，再在这里填路径
# 不设或留空则从 config 的 pos_dir/neg_dir 读（小数据、全量进内存）
TRAIN_FILELIST="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_dense2b_txt/qwen3_sft_itn/train.jsonl.test.filelist"   # 例如: output/keyword_corrector3_jsonl/train.jsonl.filelist
EVAL_FILELIST="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_dense2b_txt/qwen3_sft_itn/dev.jsonl.test.filelist"    # 例如: output/keyword_corrector3_jsonl/dev.jsonl.filelist

train_args="--resume --model_dir $dir"
# train_args="--model_dir $dir"
if [ -n "$TRAIN_FILELIST" ] && [ -f "$TRAIN_FILELIST" ]; then
  train_args="$train_args --train_filelist $TRAIN_FILELIST"
  [ -n "$EVAL_FILELIST" ] && [ -f "$EVAL_FILELIST" ] && train_args="$train_args --eval_filelist $EVAL_FILELIST"
fi

if [ $stage -le 4 ]; then
  if [ "$use_hope" == "true" ]; then
    priority=P1
    launch_cmd="$AM_ROOT/kaldi_utils/parallel/hope_submit_gpu.py \
--job_name $job_name \
--timeout 72 \
--log_dir $hope_log_dir \
--gpu_type $gpu_type \
--priority $priority \
--nnodes $nnodes \
--image $image \
--nproc_per_node $nproc_per_node \
--submitter huangzijian07 \
--queue $gpu_queue"
  else
    launch_cmd="python -m torch.distributed.launch \
--nproc_per_node $nprocs \
--master_port 29003"
  fi

  set -x
  $launch_cmd train_keyword_corrector3.py $train_args || exit 1
fi
