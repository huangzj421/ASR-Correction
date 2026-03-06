from transformers import AutoTokenizer
import numpy as np
import torch

# tokenizer_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-speech-llm/hadoop-speech/production_training/model_storage/longcat_flash_iter_0214000/format_hf"
tokenizer_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/xianghongyu/y25h2/asr/tokenizer/auto_tokenizer_mask/test_save"
text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

batch_len = 2000
context_len = 8192 + 1

t_unk = 0
t_end = 2
t_pad = 3

def get_text_from_npy(npy_file_path):
    data = np.load(npy_file_path)
    data = torch.from_numpy(data)
    text_list = []
    for sentence in data:
        indices = (sentence == t_end).nonzero(as_tuple=True)[0]
        last_pos = 0
        for pos in indices:
            # 不要 pos，是结束符
            text = sentence[last_pos:pos].tolist()
            text = text_tokenizer.decode(text)
            text_list.append(text)
            last_pos = pos + 1
    return text_list

if __name__ == "__main__":
    # npy_file_path = "/mnt/hdfs/user/hadoop-llm-data/ai-data/data-cube/release/515/tokenize_merge/llm-pretrain-corpus/zh_merge_half1_datasets/data/shard-0000-2000-part-0.npy"
    npy_file_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/xianghongyu/y25h2/asr/llm_itn/data/data/0/0.npy"
    text_list = get_text_from_npy(npy_file_path)
    import pdb; pdb.set_trace()
    # 实行包案责任制挂图作战及时督促<mask_0>实行包案责任制，挂图作战及时督促