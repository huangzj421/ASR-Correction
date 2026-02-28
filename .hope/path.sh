set -x
# set -e
conda env list
conda list
which conda
which python
which python3
ls /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/anaconda3/envs/correction/bin
echo $PYTHONPATH
export PYTHONPATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/anaconda3/envs/correction/bin:$PYTHONPATH
echo $PYTHONPATH
echo $PATH
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/anaconda3/envs/correction/bin:$PATH
echo $PATH
which conda
which python
which python3

# 创建一个临时Python脚本来测试导入kaldiio
TEMP_SCRIPT=$(mktemp /tmp/test_transformers_import.XXXXXX.py)

cat <<EOF > $TEMP_SCRIPT
try:
    import transformers
    print("成功导入transformers")
except ImportError as e:
    print("导入transformers失败:", e)
EOF

# 运行临时Python脚本
python $TEMP_SCRIPT