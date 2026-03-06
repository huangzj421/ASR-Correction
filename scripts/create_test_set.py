import json
import re
import os
from pathlib import Path
from typing import List, Set, Tuple, Dict
import wordfreq
import jieba
segmenter = jieba
import math

# ============================================================================
# 停用词加载（从 split_context_keywords.py 拷贝）
# ============================================================================
STOPWORDS_FILES = [
    Path(__file__).parent / 'stop_words1.txt',
    Path(__file__).parent / 'stop_words2.txt',
    Path(__file__).parent / 'stop_words3.txt',
]

_STOP_WORDS = None

def load_stopwords_from_local():
    """从本地停用词文件加载（读取三个文件并合并）"""
    stopwords_set = set()
    loaded_files = 0
    
    for file_path in STOPWORDS_FILES:
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords_set.add(word)
            loaded_files += 1
        except Exception:
            continue
    
    if loaded_files > 0:
        return stopwords_set
    return None

def get_stopwords():
    """获取停用词集合（懒加载）"""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = load_stopwords_from_local()
        if _STOP_WORDS is None:
            _STOP_WORDS = set()
    return _STOP_WORDS


# ============================================================================
# 分词和关键词提取（从 split_context_keywords.py 拷贝）
# ============================================================================
def split_context_keywords(text: str) -> List[str]:
    """从文本中提取关键词
    
    按照空格分割文本，对于每个词：
    - 长度 < 2: 跳过
    - 长度 <= 5: 直接加入关键词（过滤停用词）
    - 长度 > 5: 使用分词工具分词，提取长度 >= 2 的子词（过滤停用词）
    """
    if not text or not isinstance(text, str):
        return []
    
    stop_words = get_stopwords()
    words = text.split()
    keywords = set()

    for word in words:
        word = word.strip()
        if len(word) < 2:
            continue
        elif len(word) <= 5:
            if word not in stop_words:
                keywords.add(word)
        else:
            # 使用分词工具进行分词
            sub_words = segmenter.cut(word)

            for sub_word in sub_words:
                sub_word = sub_word.strip()
                if len(sub_word) >= 2 and sub_word not in stop_words:
                    keywords.add(sub_word)
    
    return sorted(keywords)


def segment_text(text: str) -> List[str]:
    """对文本进行分词"""
    try:
        words = list(segmenter.cut(text))
        # 过滤空字符串和标点
        words = [w.strip() for w in words if w.strip() and not re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', w)]
        return words
    except:
        # 简单分词：按空格和标点分割
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
        return words


# ============================================================================
# 编辑距离计算（从 select_keywords_for_asr_correction.py 拷贝）
# ============================================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein距离）"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """计算归一化的编辑距离（0-1之间，越小越相似）"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    distance = levenshtein_distance(s1, s2)
    return distance / max_len


# ============================================================================
# 拼音相似度计算（从 select_keywords_for_asr_correction.py 拷贝）
# ============================================================================
def get_pinyin(text: str) -> str:
    """将中文文本转换为拼音"""
    try:
        from pypinyin import lazy_pinyin, Style
        pinyin_list = lazy_pinyin(text, style=Style.NORMAL)
        return ''.join(pinyin_list)
    except ImportError:
        return text


def compute_edit_distance_similarity(candidate: str, vocab_word: str) -> float:
    """计算词级编辑距离相似度"""
    distance = normalized_edit_distance(candidate, vocab_word)
    return 1.0 - distance


def compute_phoneme_distance_similarity(candidate: str, vocab_word: str) -> float:
    """计算音素编辑距离相似度（基于拼音）"""
    try:
        pinyin1 = get_pinyin(candidate)
        pinyin2 = get_pinyin(vocab_word)
        distance = normalized_edit_distance(pinyin1, pinyin2)
        return 1.0 - distance
    except Exception:
        return 0.0


# ============================================================================
# 匹配逻辑（从 rare_words_matcher.py 拷贝）
# ============================================================================
def filter_stopwords(words: List[str], stopwords: Set[str]) -> List[str]:
    """从词列表中移除停用词"""
    filtered = [w for w in words if w.strip() and w not in stopwords]
    return filtered


def generate_sub_combinations(text: str, min_len: int = 2, max_len: int = 5) -> List[str]:
    """生成文本的所有可能的子组合（连续子串）"""
    sub_combinations = []
    text_len = len(text)
    
    for start in range(text_len):
        for end in range(start + min_len, min(start + max_len + 1, text_len + 1)):
            sub_str = text[start:end]
            if sub_str.strip():
                sub_combinations.append(sub_str)
    
    return sub_combinations


def match_candidate_to_vocab(
    candidate: str,
    vocab_list: List[str],
    top_k: int = 10,
    weights: Dict[str, float] = None
) -> List[Tuple[str, float]]:
    """
    将候选词与词汇列表进行匹配，返回 Top-K 最相关的词汇
    
    Args:
        candidate: 候选词
        vocab_list: 词汇列表
        top_k: 返回前k个
        weights: 各指标的权重
    
    Returns:
        List[Tuple[词汇, 最终分数]]
    """
    if weights is None:
        weights = {
            'edit': 0.4,
            'phoneme': 0.5
        }
    
    results = []
    
    # 计算每个词汇的相似度
    for vocab_word in vocab_list:
        # 1. 词级编辑距离相似度
        edit_score = compute_edit_distance_similarity(candidate, vocab_word)
        
        # 2. 音素编辑距离相似度
        phoneme_score = compute_phoneme_distance_similarity(candidate, vocab_word)
        
        # 计算加权最终分数
        final_score = (
            edit_score * weights['edit'] +
            phoneme_score * weights['phoneme']
        )
        
        # 归一化
        total_weight = sum(w for w in weights.values() if w > 0)
        if total_weight > 0:
            final_score = final_score / total_weight
        
        results.append((vocab_word, final_score))
    
    # 按分数排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 Top-K
    return results[:top_k]

def _is_chinese(text: str) -> bool:
    """判断文本是否包含中文字符"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def _word_importance_score(word: str, lang: str = 'zh') -> float:
    """
    基于 wordfreq 计算词语的重要性分数（对数逆频率，越高越稀有）。
    参考 select_low_freq_keywords.get_word_importance。
    """
    if wordfreq is None:
        return 0.0
    word = word.strip()
    if not word:
        return 0.0
    epsilon = 1e-10
    freq = wordfreq.word_frequency(word, lang)
    if freq < 1e-8:
        if lang == 'zh':
            segments = list(segmenter.cut(word))
            segments = [s.strip() for s in segments if s.strip()]
            if len(segments) > 1:
                seg_freqs = [wordfreq.word_frequency(s, 'zh') for s in segments]
                freq = min(seg_freqs) if seg_freqs else epsilon
            else:
                freq = epsilon
        else:
            freq = epsilon
    return -math.log(freq + epsilon)


def _get_top_k_low_freq_words(keywords: str, vocab_list: List[str], top_k: int) -> str:
    """当 hyp 无法产生任何子组合时，用 wordfreq 从 vocab 中选出 top k 个低频词（对数逆频率）。"""
    if not vocab_list or top_k <= 0:
        return ""
    if wordfreq is None:
        # 未安装 wordfreq 时退化为按在 keywords 中出现次数最少
        if not keywords:
            return ' '.join(sorted(vocab_list[:top_k]))
        freq_list = []
        for word in vocab_list:
            w = word.strip()
            if not w:
                continue
            freq_list.append((w, keywords.count(w)))
        freq_list.sort(key=lambda x: (x[1], x[0]))
        top_words = [w for w, _ in freq_list[:top_k]]
        return ' '.join(sorted(top_words))
    # 使用 wordfreq：按重要性分数降序（分数越高越稀有），取 top_k
    scored = []
    for word in vocab_list:
        w = word.strip()
        if not w:
            continue
        lang = 'zh' if _is_chinese(w) else 'en'
        score = _word_importance_score(w, lang)
        scored.append((w, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in scored[:top_k]]
    return sorted(top_words)

def clean_text(keywords: str, hyp: str, min_sub_len: int = 2, max_sub_len: int = 5, max_output_words: int = 100) -> str:
    """
    清理文本并匹配罕见词
    
    Args:
        keywords: keywords 字段的原始文本
        hyp: ASR转录文本
        min_sub_len: 子组合最小长度（默认2）
        max_sub_len: 子组合最大长度（默认5）
        max_output_words: 最大输出词汇数量（默认100）
    
    Returns:
        匹配的罕见词列表（空格分隔的字符串），最多 max_output_words 个词
    """
    # 步骤1: 从 keywords 字段提取词汇列表
    vocab_list = split_context_keywords(keywords)

    # 步骤2: 对 hyp 进行分词
    words = segment_text(hyp)
    
    if not vocab_list:
        stopwords = get_stopwords()
        fallback = filter_stopwords(words, stopwords) if words else words
        candidates = fallback if fallback else words
        seen = set()
        out = []
        for w in candidates:
            w = w.strip()
            if w and len(w) >= 2 and w not in seen:
                seen.add(w)
                out.append(w)
                if len(out) >= max_output_words:
                    break
        return sorted(out)
    
    # 步骤3: 过滤停用词，得到罕见词候选
    stopwords = get_stopwords()
    rare_candidates = filter_stopwords(words, stopwords)
    
    # if not rare_candidates:
    #     return []
    candidates = rare_candidates if rare_candidates else words
    
    # 步骤4: 生成所有可能的子组合
    all_sub_combinations = []
    # for candidate in rare_candidates:
    for candidate in candidates:
        sub_combs = generate_sub_combinations(candidate, min_sub_len, max_sub_len)
        all_sub_combinations.extend(sub_combs)
    
    # 去重
    all_sub_combinations = list(set(all_sub_combinations))
    
    if not all_sub_combinations:
        # import pdb; pdb.set_trace()
        return _get_top_k_low_freq_words(keywords, vocab_list, max_output_words)
    
    # 步骤5: 根据候选词数量动态计算每个候选词的 top_k
    # 目标是：每个候选词匹配 max_output_words // 候选词数量 个词汇
    # 但要设置一个合理的上限（例如50）和下限（至少1个）
    num_candidates = len(all_sub_combinations)
    dynamic_top_k = max(1, min(max_output_words // num_candidates, 50))
    
    # 步骤6: 为每个候选词匹配 Top-K 最相关的词汇（保留分数信息）
    # 使用字典存储每个词汇及其最高分数
    word_scores = {}  # {word: max_score}
    
    for candidate in all_sub_combinations:
        matches = match_candidate_to_vocab(
            candidate=candidate,
            vocab_list=vocab_list,
            top_k=dynamic_top_k
        )
        
        # 收集所有匹配的词汇和分数，对于重复的词汇保留最高分数
        for vocab_word, score in matches:
            vocab_word_clean = vocab_word.strip()
            if vocab_word_clean:
                if vocab_word_clean not in word_scores:
                    word_scores[vocab_word_clean] = score
                else:
                    # 保留最高分数
                    word_scores[vocab_word_clean] = max(word_scores[vocab_word_clean], score)
    
    # # 步骤7: 将 rare_candidates 也加入到结果中
    # # 给它们一个较高的默认分数（0.8），确保它们会被优先保留
    # # 如果它们已经在 word_scores 中，保留已有的分数（通常是匹配得到的更高分数）
    # for rare_candidate in candidates:
    #     rare_candidate_clean = rare_candidate.strip()
    #     if rare_candidate_clean and len(rare_candidate_clean) >= 2:
    #         if rare_candidate_clean not in word_scores:
    #             word_scores[rare_candidate_clean] = 0.8  # 默认分数
    #         # 如果已经存在，保留已有的分数（通常是匹配得到的更高分数）
    
    # 步骤8: 如果结果数量 <= max_output_words，直接返回
    if len(word_scores) <= max_output_words:
        unique_matched_words = sorted(word_scores.keys())
        return unique_matched_words
    
    # 步骤9: 如果结果数量 > max_output_words，按分数排序取前 max_output_words 个
    # 按分数降序排序，然后取前 max_output_words 个
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in sorted_words[:max_output_words]]
    top_words_sorted = sorted(top_words)  # 按字母顺序排序
    
    return top_words_sorted


if __name__ == "__main__":
    jsonl_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/ASR_keywords_offline/beam2000/ref_hyp_context_online.jsonl'
    output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/huangzijian07/data/longcat-s/train/prepare/asr_correction/data_distract_no_homo/test_set'
    special_token = '<mask_99>'
    with open(jsonl_path, 'r', encoding='utf-8') as f, open(f'{output_path}/refs.txt', 'w', encoding='utf-8') as ref_f, open(f'{output_path}/hyps_before.txt', 'w', encoding='utf-8') as hyp_f, open(f'{output_path}/prompt.txt', 'w', encoding='utf-8') as prompt_f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            data = json.loads(line)
            ref = data['text']
            hyp = data['hyp']
            # keywords = data['keywords'].strip().replace(" ", ", ")
            keywords = data['keywords']
            filter_keywords = clean_text(keywords, hyp, max_output_words=10)
            prompt = f"{hyp}{special_token}{', '.join(filter_keywords)}{special_token}"
            ref_f.write(str(idx) + ' ' + ref + '\n')
            hyp_f.write(str(idx) + ' ' + hyp + '\n')
            prompt_f.write(str(idx) + ' ' + prompt + '\n')