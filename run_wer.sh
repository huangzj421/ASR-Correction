WER=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/xianghongyu/code/speech/bazel-bin/asr/wer/wer
# REF=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/text2
REF=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/refs_100.txt
input_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/ASR-Correction/output/model_qwen3_keyword_corrector3/infer_output_filter_100

for file in $input_path/*.txt; do
    echo "Processing $file"
    $WER $REF $file ${file%.txt}_wer.log
done