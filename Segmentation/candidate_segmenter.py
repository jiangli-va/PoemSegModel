# poem_llm_segmenter/candidate_segmenter.py
# -*- coding: utf-8 -*-
"""
候选分词生成：
- 对一行诗（去标点后）枚举所有可能切分（限制最大词长）
- 对每种切分用：词典得分 + 音步得分 打分
- 返回 top-K 个候选，用于后续 LLM 选择
"""

import re
from typing import List, Tuple, Dict, Optional

from config import load_vocab_and_scores


_PUNCT_PATTERN = re.compile(r"[，。！？、；：,.!?\s“”\"‘’（）\(\)\[\]【】《》〈〉…]+")



def strip_punct(text: str) -> str:
    """去掉常见中文标点与空白，只保留汉字等，用于统计字数。"""
    return _PUNCT_PATTERN.sub("", text)


def detect_line_type(line: str) -> Optional[str]:
    """
    判断一行是否五言 / 七言：
      返回 '5' / '7' / None
    """
    pure = strip_punct(line)
    n = len(pure)
    if n == 5:
        return "5"
    elif n == 7:
        return "7"
    else:
        return None


def rhythm_bonus(
    start: int,
    end: int,
    total_len: int,
    line_type: Optional[str],
) -> float:
    """
    音步奖励函数：
      - 五言：鼓励在 2 的位置有词边界（2+3），惩罚在3的位置有词边界
      - 七言：鼓励在 2、4 的位置有词边界（2+2+3）；惩罚在3、5的位置有词边界
      - 对 1 字词略微惩罚，对 2、3 字词略微奖励
    """
    L = end - start
    bonus = 0.0

    if line_type == "5":
        if end == 2:
            bonus += 0.8    # 原为0.8
        elif end == 3:
            bonus -= 0.3    # 新增惩罚
        if L == 1:
            bonus -= 0.05    # 原为0.3【确认下权重】
        elif L == 2:
            bonus += 0.1
    elif line_type == "7":
        if end in (2, 4):
            bonus += 0.5    # 原为0.6
        elif end in (3, 5):
            bonus -= 0.3    # 新增惩罚
        if L == 1:
            bonus -= 0.05   # 原为0.25，减少一字词的惩罚比例
        elif L == 2:
            bonus += 0.1
        elif L >= 4:
            bonus -= 0.1
    else:
        if L == 1:
            bonus -= 0.05    # 原为0.1，减少一字词的惩罚比例
        elif L == 2:
            bonus += 0.05

    # # 行尾用 >=2 字收尾稍微奖励【该理由不成立：句尾用一个 2 字或 3 字的自然词收尾”，往往比单字收尾更自然。】
    # if end == total_len and L >= 2:
    #     bonus += 0.05

    return bonus


def enumerate_segmentations(
    text: str,
    vocab_set: set,
    word_score: Dict[str, float],
    line_type: Optional[str],
    max_word_len: int = 4,
    unk_score: float = -1.0,    # 词典中没有的词的基础分，原为-3.0
    max_paths: int = 5000,
) -> List[Tuple[List[str], float]]:
    """
    对去标点后的纯字串 text 全枚举所有切分（限制最大词长），并打分。

    返回：
      list[(tokens, score)]，按 score 从高到低排序（可能被截断到 max_paths）
    """
    n = len(text)
    results: List[Tuple[List[str], float]] = []

    def dfs(start: int, cur_tokens: List[str], cur_score: float):
        if len(results) >= max_paths:
            return

        if start == n:
            results.append((cur_tokens[:], cur_score))
            return

        # 从 start 出发，尝试 1~max_word_len 字
        for L in range(1, max_word_len + 1):
            end = start + L
            if end > n:
                break
            w = text[start:end]

            base = word_score.get(w, unk_score)
            if w not in vocab_set:
                base -= 0.1  # 不在词表略减分，原为0.5

            score = cur_score + base + rhythm_bonus(start, end, n, line_type)


            cur_tokens.append(w)
            dfs(end, cur_tokens, score)
            cur_tokens.pop()

    dfs(0, [], 0.0)

    # 按得分排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_paths]


def generate_candidates_for_line(
    line: str,
    vocab: List[str],
    word_score: Dict[str, float],
    max_word_len: int = 4,
    top_k: int = 5,
) -> List[Tuple[List[str], float]]:
    """
    对一行原始诗句生成 top-K 候选分词。

    返回：
      list[(tokens, score)]，长度 <= top_k
    """
    pure = strip_punct(line)
    if not pure:
        return []

    line_type = detect_line_type(line)
    vocab_set = set(vocab)

    # 动态枚举切分并打分
    all_segs = enumerate_segmentations(
        text=pure,
        vocab_set=vocab_set,
        word_score=word_score,
        line_type=line_type,
        max_word_len=max_word_len,
        max_paths=5000,
    )

    # ======= 新增：最大匹配奖励 ========（169-294行）
    # 1、加载自定义词典
    custom_dict_path = './dict_all.txt'
    custom_trie = {}  
    try:
        with open(custom_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    node = custom_trie
                    for char in word:
                        node = node.setdefault(char, {})
                    node['<END>'] = True  # 标记词尾
    except FileNotFoundError:
        print(f"[注意] 未找到自定义词典文件，跳过最大匹配奖励: {custom_dict_path}")
        custom_trie = None

    # 2、定义最大匹配函数
    def max_match_cut(text, trie, mode='forward'):
        if not text or not trie:
            return []
        tokens = []
        if mode == 'forward':
            i = 0
            while i < len(text):
                node = trie
                j = i
                last_match = i + 1  # 默认匹配单字
                while j < len(text) and text[j] in node:
                    node = node[text[j]]
                    j += 1
                    if '<END>' in node:
                        last_match = j
                tokens.append(text[i:last_match])
                i = last_match
        else:  # backward
            i = len(text)
            while i > 0:
                node = trie
                j = i - 1
                last_match = i - 1
                while j >= 0 and text[j] in node:
                    node = node[text[j]]
                    if '<END>' in node:
                        last_match = j
                    j -= 1
                tokens.insert(0, text[last_match:i])
                i = last_match
        return tokens
    
    # 3、若成功加载自定义词典，则进行正逆向最大匹配
    mm_results_for_reward = []
    if custom_trie:
        fmm_tokens = max_match_cut(pure, custom_trie, mode='forward')
        bmm_tokens = max_match_cut(pure, custom_trie, mode='backward')
        # 去重
        seen = set()
        for tokens in [fmm_tokens, bmm_tokens]:
            if tokens:
                token_str = '|'.join(tokens)
                if token_str not in seen:
                    mm_results_for_reward.append(tokens)
                    seen.add(token_str)

    # 4、对与匹配结果相同的DP候选增加奖励
    if mm_results_for_reward:
        rewarded_segs = []
        for tokens, dp_score in all_segs:  # 遍历原有DP结果
            final_score = dp_score
            # 检查当前 tokens 是否与任一 MM 结果完全一致
            for mm_tokens in mm_results_for_reward:
                if tokens == mm_tokens:
                    # 完全匹配，给予较高奖励（例如 10 分）
                    final_score += len(mm_tokens) * 2    # 依据词数给予奖励
                    break  
            rewarded_segs.append((tokens, final_score))
        # 按新分数重新排序
        rewarded_segs.sort(key=lambda x: x[1], reverse=True)
        # 返回 top_k
        return rewarded_segs[:top_k]
    # ====== 新增部分结束 ======

    return all_segs[:top_k]


def format_tokens(tokens: List[str]) -> str:
    """把 tokens 格式化成 用 | 分隔 的字符串，便于喂给 LLM 和保存。"""
    return "|".join(tokens)


if __name__ == "__main__":
    # 简单测试
    vocab, word_score = load_vocab_and_scores()
    line = "空山新雨后"
    cands = generate_candidates_for_line(line, vocab, word_score, max_word_len=4, top_k=5)
    print("原句：", line)
    for i, (tokens, score) in enumerate(cands):
        print(f"候选 {i}: {format_tokens(tokens)}  (score={score:.3f})")
