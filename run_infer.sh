python run_infer.py \
  --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt2_2 \
  --model_dir output/model_qwen3_keyword_corrector3_dense2b_text/ \
  --output_dir output/model_qwen3_keyword_corrector3_dense2b_text/infer_output \

# python run_infer.py \
#   --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt_filter_100.txt \
#   --model_dir output/model_qwen3_keyword_corrector3_all/ \
#   --output_dir output/model_qwen3_keyword_corrector3_all/infer_output_filter_100 \
#   --max_length 1024 \
#   --batch_size 1

# python run_infer.py \
#   --input_txt /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt.txt \
#   --model_dir output/model_qwen3_keyword_corrector3_all/ \
#   --output_dir output/model_qwen3_keyword_corrector3_all/infer_output_no_filter \
#   --max_length 1024 \
#   --batch_size 1