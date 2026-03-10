#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入 ref, hyp1, hyp2 三个文本文件，每行格式为「序号 text」。
对每行按序号匹配，计算 ref-hyp1 的 wer1、ref-hyp2 的 wer2，
用 jiwer.visualize_alignment 输出对比结果，
最后按 wer1 降序、wer2 升序排序输出。
独立运行，仅依赖 jiwer。
"""

import argparse
import re
import sys
import json
import jiwer


class RemovePunctuation(jiwer.transforms.AbstractTransform):
    def process_string(self, s: str):
        s = s.replace('-', ' ')
        s = s.replace('.', ' ')
        s = s.replace(',', ' ')
        s = s.replace('?', ' ')
        s = s.replace('、', ' ')
        s = s.replace('，', ' ')
        s = s.replace('。', ' ')
        s = s.replace('？', ' ')
        s = s.replace('！', ' ')
        s = s.replace('[UNK]', ' ')
        # for c in "呃啊呀嘞哎哦啦嗯喂了":
        #     s = s.replace(c, '')
        return s


class ChineseEnglishSplitter(jiwer.transforms.AbstractTransform):
    """将文本切分为词，非英文的每一个字符算一个词，英文和数字按照空格分为词"""

    def process_string(self, s: str):
        result = []
        pattern = re.compile(r"([a-zA-Z0-9']+)|([^a-zA-Z0-9'])")
        for m in pattern.findall(s):
            if m[0]:
                result.append(m[0])
            elif m[1]:
                for c in m[1]:
                    if c.strip():
                        result.append(c)
        return " ".join(result)


wer_tr = jiwer.transforms.Compose([
    jiwer.transforms.ToUpperCase(),
    RemovePunctuation(),
    ChineseEnglishSplitter(),
    jiwer.transforms.ReduceToListOfListOfWords(),
])


def cal_wer(refs, hyps):
    word_output = jiwer.process_words(
        reference=refs,
        hypothesis=hyps,
        reference_transform=wer_tr,
        hypothesis_transform=wer_tr,
    )
    wer = word_output.wer
    e_sub = word_output.substitutions
    e_ins = word_output.insertions
    e_del = word_output.deletions
    hits = word_output.hits
    ref_len = hits + e_sub + e_del
    hyp_len = hits + e_sub + e_ins
    return ref_len, hyp_len, e_sub, e_ins, e_del, wer, word_output


def load_key_value(fn):
    """每行 key value，tab 转空格，按第一个空格拆成 key 与 value；value 为 '.' 的跳过。"""
    res = {}
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            content = line.replace("\t", " ").strip("\n").strip().split(" ", 1)
            if len(content) == 2:
                key, value = content
            else:
                key = content[0]
                value = ""
            if value == ".":
                continue
            res[key] = value
    return res

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def main():
    parser = argparse.ArgumentParser(
        description="输入 ref, hyp1, hyp2 三个文本（每行 key value），输出 alignment 对比，按 wer1 降序、wer2 升序")
    parser.add_argument("ref", help="参考文本文件，每行: key value")
    parser.add_argument("hyp1", help="假设1文本文件，每行: key value")
    parser.add_argument("hyp2", help="假设2文本文件，每行: key value")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径，默认 stdout")
    args = parser.parse_args()

    ref_dict = load_key_value(args.ref)
    hyp1_dict = load_key_value(args.hyp1)
    hyp2_dict = load_key_value(args.hyp2)

    jsonl_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/test_set/prompt.txt'

    data_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            data = line.split(" ", 1)[1]
            ref = data.split("<mask_99>")[0]
            keywords = data.split("<mask_99>")[1]
            data_dict[ref] = keywords

    common_keys = sorted(set(ref_dict) & set(hyp1_dict) & set(hyp2_dict))
    if not common_keys:
        print("错误: 没有找到 ref/hyp1/hyp2 共有的 key", file=sys.stderr)
        sys.exit(1)

    # 对每条计算 wer1(ref,hyp1), wer2(ref,hyp2)，单句 WER = wer/ref_len（ref_len>0 否则 1）；变换后为空的跳过
    rows = []
    model_names = ["hyp1", "hyp2"]
    for key in common_keys:
        ref_text = ref_dict[key]
        h1 = hyp1_dict[key]
        h2 = hyp2_dict[key]
        if remove_punctuation(h1) == remove_punctuation(h2):
            continue
        try:
            ref_len1, _, _, _, _, wer1_raw, word_out1 = cal_wer([ref_text], [h1])
            ref_len2, _, _, _, _, wer2_raw, word_out2 = cal_wer([ref_text], [h2])
        except ValueError:
            continue
        wer1 = wer1_raw / ref_len1 if ref_len1 > 0 else 1.0
        wer2 = wer2_raw / ref_len2 if ref_len2 > 0 else 1.0

        align1 = jiwer.visualize_alignment(word_out1, skip_correct=False)
        align2 = jiwer.visualize_alignment(word_out2, skip_correct=False)
        lines1 = align1.split("\n")
        lines2 = align2.split("\n")
        ref_line = lines1[2].replace("*", "* ") if len(lines1) > 2 else ref_text
        hyp1_line = lines1[3].replace("*", "* ") if len(lines1) > 3 else h1
        hyp2_line = lines2[3].replace("*", "* ") if len(lines2) > 3 else h2
        rows.append({
            "key": key,
            "wer1": wer1,
            "wer2": wer2,
            "ref_line": ref_line,
            "hyp1_line": hyp1_line,
            "hyp2_line": hyp2_line,
            "ref_text": ref_text,
            "hyp1_text": h1,
            "hyp2_text": h2,
            "keywords": data_dict.get(h1, [])
        })

    # 排序：wer1 降序，wer2 升序
    rows.sort(key=lambda r: (-r["wer1"], r["wer2"]))

    out = sys.stdout if args.output is None else open(args.output, "w", encoding="utf-8")
    try:
        for r in rows:
            out.write(f"{r['key']} {r['wer1']:.4f}    Keywords: {r['keywords']}\n")
            out.write(f"{' '*25} REF: {r['ref_text']}\n")
            wer_str = str(float(round(r["wer1"], 4)))
            out.write(f"{' '*(18-len('hyp1'))}hyp1|{wer_str + '0' * (6 - len(wer_str))} HYP: {r['hyp1_text']}\n")
            wer_str = str(float(round(r["wer2"], 4)))
            out.write(f"{' '*(18-len('hyp2'))}hyp2|{wer_str + '0' * (6 - len(wer_str))} HYP: {r['hyp2_text']}\n")
    finally:
        if args.output is not None:
            out.close()

    if args.output:
        print(f"已写入 {args.output}，共 {len(rows)} 条", file=sys.stderr)


if __name__ == '__main__':
    main()
