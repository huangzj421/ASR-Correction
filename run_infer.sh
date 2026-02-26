# python run_infer.py \
#   --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt2_2 \
#   --model_dir output/model_qwen3_keyword_corrector3/ \
#   --output_dir output/model_qwen3_keyword_corrector3/infer_output \
#   --max_length 1024 \
#   --batch_size 16

# python run_infer.py \
#   --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt_filter_100.txt \
#   --model_dir output/model_qwen3_keyword_corrector3/ \
#   --output_dir output/model_qwen3_keyword_corrector3/infer_output_filter_100 \
#   --max_length 1024 \
#   --batch_size 16

python run_infer.py \
  --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt.txt \
  --model_dir output/model_qwen3_keyword_corrector3/checkpoint-12000 \
  --output_dir output/model_qwen3_keyword_corrector3/infer_output_beam2000 \
  --max_length 1024 \
  --batch_size 16