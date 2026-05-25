# poem_llm_segmenter/config.py
# -*- coding: utf-8 -*-
"""
配置 & 词表加载：
- 复用你的 utils_trie.read_vocab
- 从 Excel 中构造 word_score（词 -> 得分）
"""

import os
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd

import utils_trie  # 确保在 PYTHONPATH 或同目录


# DEFAULT_VOCAB_PATH = os.path.join("词表汇总", "古诗分词用词典v1.xlsx")
DEFAULT_VOCAB_PATH = './古汉语词典及词频5以上bigram总结表v9.xlsx'


def build_word_score_dict(
    vocab_path: str,
    min_length: int = 1,
    max_length: int = -1,
) -> Dict[str, float]:
    """
    从古诗专用词表中构造：词 -> 得分 的字典。

    评分大致考虑：
      - 频度（优先用“调整后总频度”，其次“子语料总频度”，否则“频度”）
      - MI（互信息）
      - 包含词典数（出现在越多辞书越好）
      - 是否典故（“典故词典岂”“搜韵典故”）
      - 词长偏好（古诗偏好 2 字，其次 3 字，4 字以上更谨慎）
    """
    df = pd.read_excel(vocab_path, sheet_name="Sheet1")

    if "长度" not in df.columns:
        raise ValueError("词表中缺少“长度”列，请检查 Excel 列名。")

    df["长度"] = df["长度"].astype(int)
    if max_length > 0:
        df = df[(df["长度"] >= min_length) & (df["长度"] <= max_length)]
    else:
        df = df[df["长度"] >= min_length]

    def pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    freq_col = pick_col(["调整后总频度", "子语料总频度", "频度"])
    if freq_col is None:
        raise ValueError("词表中找不到任何频度列：['调整后总频度', '子语料总频度', '频度']")

    mi_col = "MI" if "MI" in df.columns else None
    dict_cnt_col = "包含词典数" if "包含词典数" in df.columns else None
    diangu_cols = [c for c in ["典故词典岂", "搜韵典故"] if c in df.columns]

    # 填充缺失值
    df[freq_col] = df[freq_col].fillna(0).astype(float)
    if mi_col:
        df[mi_col] = df[mi_col].fillna(0).astype(float)
        mi_min = float(df[mi_col].min())
        mi_max = float(df[mi_col].max())
    else:
        mi_min = mi_max = 0.0

    if dict_cnt_col:
        df[dict_cnt_col] = df[dict_cnt_col].fillna(0).astype(float)

    word_score: Dict[str, float] = {}

    for _, row in df.iterrows():
        w = str(row["词条"])
        L = int(row["长度"])

        # 基础分：log(频度+1)
        freq_val = float(row[freq_col]) if not pd.isna(row[freq_col]) else 0.0
        score = math.log1p(freq_val)

        # MI 归一化
        if mi_col and mi_max > mi_min:
            mi_val = float(row[mi_col]) if not pd.isna(row[mi_col]) else 0.0
            mi_norm = (mi_val - mi_min) / (mi_max - mi_min + 1e-6)
            score += 0.8 * mi_norm

        # 包含词典数
        if dict_cnt_col:
            cnt_val = float(row[dict_cnt_col]) if not pd.isna(row[dict_cnt_col]) else 0.0
            score += 0.3 * cnt_val

        # 是否典故
        for c in diangu_cols:
            v = row[c]
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if v > 0:
                score += 1.0
                break

        # 词长偏好
        if L == 2:
            score += 0.5
        elif L == 3:
            score += 0.2
        elif L >= 4:
            score -= 0.2

        word_score[w] = float(score)

    return word_score


def load_vocab_and_scores(
    vocab_path: Optional[str] = None,
    min_length: int = 1,
    max_length: int = 4,
) -> Tuple[List[str], Dict[str, float]]:
    """
    一站式加载：
      - 从 Excel 读出完整词表
      - 构造词得分字典（限制到 <=4 字）
    """
    if vocab_path is None:
        vocab_path = DEFAULT_VOCAB_PATH

    # 复用你现有的 read_vocab：max_length<=0 时返回全词表
    vocab, _ = utils_trie.read_vocab(vocab_path=vocab_path, min_length=-1)

    word_score = build_word_score_dict(
        vocab_path=vocab_path,
        min_length=min_length,
        max_length=max_length,
    )
    return vocab, word_score
