#!/bin/bash
# 使用 Hugging Face 上的 twnlp/ChineseErrorCorrector3-4B 对测试集纠错
# 输入格式：每行「序号 错误文本<mask_99>关键词<mask_99>」
# 输出：每行「序号 纠错后的文本」到 <input_basename>_cec3.txt

INPUT_TXT="${1:-prompt.txt}"   # 默认 prompt.txt，也可传参

python run_infer_cec3.py \
  --input_txt "$INPUT_TXT" \
  --max_new_tokens 512 \
  --batch_size 1

# 指定输出文件示例：
# python run_infer_cec3.py --input_txt prompt.txt --output_txt result_cec3.txt --batch_size 8

# 使用本地已下载的模型（例如离线环境）：
# python run_infer_cec3.py --input_txt prompt.txt --model_name /path/to/ChineseErrorCorrector3-4B
