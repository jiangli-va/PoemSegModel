# llm_rerank.py
# -*- coding: utf-8 -*-
"""
让大模型在已有候选分词中，按“语法型分词（G-Seg）”规范选择最优方案。

调用方式（在 run_llm_segmentation.py 中）示例：
    from llm_rerank import choose_best_segmentation

    best_tokens, best_index = choose_best_segmentation(
        line,
        candidates,          # List[List[str]]
        model="gemini-3-pro-preview-thinking"     
    )

返回：
    best_tokens : List[str]，被选中的分词序列
    best_index  : int，候选列表中的下标（0 对应 A，1 对应 B，依此类推）
"""

from __future__ import annotations
import os
import sys
import time
from typing import List, Optional

try:
    # openai>=1.0 写法
    from openai import OpenAI
except ImportError:
    OpenAI = None


LETTER_LIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ==================== 基础：client 获取 ====================

def get_client():
    """
    根据环境变量构造 OpenAI 兼容 client。
    需要：
        OPENAI_API_KEY   （必需）
        OPENAI_API_BASE  （可选，THUNLP/自建网关时使用）
    """
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先 `pip install openai>=1.0.0`。")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY。")

    api_base = os.getenv("OPENAI_API_BASE", "").strip()
    if api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        client = OpenAI(api_key=api_key)
    return client


# ==================== Prompt 构造 ====================

def _format_candidates_for_prompt(cand_strs: List[str]) -> str:
    """
    把候选分词序列格式化为：
    A. 春风|又绿|江南|岸
    B. 春风|又|绿|江南|岸
    ...
    """
    lines = []
    for i, seg in enumerate(cand_strs):
        if i >= len(LETTER_LIST):
            break
        letter = LETTER_LIST[i]
        lines.append(f"{letter}. {seg}")
    return "\n".join(lines)


def build_rerank_prompt(line: str, cand_strs: List[str]) -> str:
    """
    构造让大模型“在候选中选”的提示词。
    重点强调：G-Seg 规范 + 功能词单立 + 只输出字母编号。
    """
    # 几个 few-shot 示例（手工写死），引导模型靠近你现在的标准答案风格
    examples = """
【示例一】
原句：春风又绿江南岸
候选：
A. 春风|又绿|江南|岸
B. 春风|又绿江|南岸
C. 春风|又|绿|江南|岸
最优答案：C

【示例二】
原句：或作金兮或似雪
候选：
A. 或作|金兮|或似雪
B. 或作|金兮或|似雪
C. 或|作|金|兮|或|似|雪
最优答案：C

【示例三】
原句：含胎十月无休歇
候选：
A. 含胎|十月|无|休歇
B. 含胎|十月|无休歇
C. 含胎|十月无|休歇
最优答案：A

【示例四】
原句：国破山河在
候选：
A. 国|破|山河|在
B. 国|破山河|在
C. 国破|山|河在
最优答案：A

【示例五】
原句：君不見灞亭耐事故將軍
候选：
A. 君不見|灞亭|耐|事故|將軍
B. 君|不見|灞亭|耐|事故|將軍
C. 君不見|灞亭|耐|事故|將|軍
最优答案：WARNING 君|不見|灞亭|耐|事|故|將軍

【示例六】
原句：黄四娘家花满蹊
候选：
A. 黄四|娘家|花|满蹊
B. 黄四|娘|家|花|满蹊
C. 黄四|娘家|花|满|蹊
最优答案：WARNING 黄四娘|家|花|满|蹊
""".strip()

    cand_block = _format_candidates_for_prompt(cand_strs)

    prompt = f"""
你是一名擅长古典汉语语法分析的分词专家，负责在多个候选分词方案中，选出最符合“语法型分词（G-Seg）”规范的一项。

【G-Seg 分词规范（请严格遵守）】
0. 总体原则：根据句意；应分尽分；实事求是；合并词、分短语。如“江南|岸”，而非“江南岸”，“楼上|雁”而非“楼上雁”，“浑|不动”而非“浑不动”。
1. **功能词、虚词（如“或、又、无、兮、其、乃、遂、也、矣”等副词、连词、语气词、否定词）应尽量单独成词。**
2. 语义紧密、习惯搭配或专有名词的词组可以合并成多字词，如“含胎”“十月”“山河”“江南”“黄阁”“东篱”“空山”“新雨”“草木”（“一” + 量词，“不” + 动词，两个近义词并列）等。
3. 有些地方的分词可能存在多种合理的方案，请你选择最符合句意表达的那一项。如“一尊|酒” “一|尊酒”都可行，则根据具体句意选择更合适的方案。
4. 不允许增删、更改任何汉字，只比较不同的切分位置；不可机械追求最长词或最多单字，每个词包含字数一般最好不超过3个。
5. **诗句里面重要的地名、人名、专有名词、固定短语，应当合并成词**，如“黄鹤楼”，“司马相如”，“鲍参军”，“韩荆州”，“长亭”，“灞桥”，“寒山寺”，“瀼西市”，“白帝城”等。

下面是若干示例，展示选择原则：

{examples}

【当前待判断句子】
原句：{line}
候选分词方案：
{cand_block}
请你先自行翻译整首诗歌，理解句意，然后根据上述 G-Seg 规范，在候选中选出最优方案，并只输出一个大写字母编号（例如：A 或 B 或 C 或 D 或 E），不要输出任何解释或其他内容。
请优先检查给出的所有候选项。如果你认为所有候选确实都有明显错误，请输出 'WARNING' + " "，后面跟着你修改后的分词结果，格式如：WARNING 春风|又|绿|江南|岸。
如果你真的要自我改动，尽量改动不超过一处，且改动后的结果不可与任何一个候选项相同。
注意，翻译需要你自行思考完成，不必输出。最终只需输出一个选项，或 WARNING 及修改结果。
""".strip()

    return prompt

# ==================== 调用大模型并解析字母 ====================

def call_llm_choose_letter(
    line: str,
    cand_strs: List[str],
    model: str = "deepseek-chat",
    max_retries: int = 50,
    sleep_sec: float = 2.0,
) -> Optional[str]:
    """
    调大模型，让它在候选中选一个字母（A/B/C/...）。
    新增：如果模型认为所有候选都不好，可能返回以 'WARNING' 开头的字符串，后面跟着它修改后的分词结果。
    返回：
        - 若选择候选：'A' / 'B' / ... 
        - 若自行修改：'WARNING|改后分词' (例如 'WARNING|春风|又|绿|江南|岸')
        - 若失败：None
    """
    client = get_client()
    prompt = build_rerank_prompt(line, cand_strs)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是擅长古典诗歌语法分析的分词判别助手，只在给定候选中选择最优分词方案。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=50,     
            )
            content = (resp.choices[0].message.content or "").strip()
            first_line = content.splitlines()[0].strip()

            print(content)
            # sys.exit(0)

            # 新增：优先检查是否为WARNING响应
            if first_line.upper().startswith("WARNING"):
                # 提取警告符号后的内容作为修改后的分词
                # 格式预期: "WARNING 春风|又|绿|江南|岸"
                warning_content = first_line[7:].strip()  # 去掉"WARNING"
                if warning_content:
                    # 统一格式，返回一个可解析的字符串
                    return f"WARNING|{warning_content}"
                else:
                    print(f"[WARN] LLM 返回了WARNING但未提供修改结果")
                    continue  # 重试

            # 提取第一个在 A..Z 范围内的字母
            for ch in first_line:
                if ch in LETTER_LIST:
                    return ch

            # 没找到合适字母，重试
            print(f"[WARN] LLM 返回内容中未找到有效字母：{repr(first_line)}")
        except Exception as e:
            print(f"[ERROR] LLM 调用失败（attempt={attempt+1}）：{e}")

        time.sleep(sleep_sec)

    # 多次重试仍失败
    return None


# ==================== 对外主函数：选择最佳分词 ====================

LETTER_LIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _extract_tokens_from_candidate(cand) -> list[str]:
    """
    尝试从 candidate 里抽出真正的“分词序列”（list[str]）。

    兼容几种常见情况：
    1) cand 本身就是 ["春风","又绿","江南","岸"]
    2) cand 是 [ ["春风","又绿","江南","岸"], score ]
    3) cand 是 {"tokens": [...], "score": ...}
    4) cand 是 "春风|又绿|江南|岸" 这种字符串
    """
    # 情况 3：字典
    if isinstance(cand, dict):
        if "tokens" in cand:
            return [str(t) for t in cand["tokens"]]
        if "seg" in cand:
            # 万一你用的是类似 {"seg": ["春风", "又绿", ...]}
            return [str(t) for t in cand["seg"]]

    # 情况 4：纯字符串
    if isinstance(cand, str):
        # 假设用 | 分隔
        parts = cand.replace(" ", "").split("|")
        return [p for p in parts if p]

    # 情况 1 / 2：list / tuple
    if isinstance(cand, (list, tuple)):
        if len(cand) == 0:
            return []
        # 若 cand[0] 还是 list/tuple，通常形如 [tokens, score]
        if isinstance(cand[0], (list, tuple)):
            # 取第一个元素作为 tokens
            tokens = cand[0]
        else:
            tokens = cand
        return [str(t) for t in tokens]

    # 其他奇怪情况，做个保守处理
    return [str(cand)]


def choose_best_segmentation(
    line: str,
    candidates,
    model: str = "deepseek-chat",
) -> tuple[list[str], int]:
    """
    在候选分词结果中，调用大模型按 G-Seg 规范选出最优的一条。
    新增：处理LLM返回的“WARNING”及自定义分词。

    参数：
        line       : 原始诗句（未分词）
        candidates : 若干候选分词结构，可能是：
                       - ["春风","又绿","江南","岸"]
                       - [ ["春风","又绿","江南","岸"], score ]
                       - {"tokens": [...], "score": ...}
                       - "春风|又绿|江南|岸"
        model      : 大模型名称

    返回：
        best_tokens : List[str]，被选中的分词序列
        best_index  : int，候选列表中的下标（0 对应 A，1 对应 B，依此类推）

    说明：
        - 若大模型调用失败或返回非法字母，则默认退回候选 0。
    """
    if not candidates:
        return [], -1

    # 先统一把所有候选转成“真正的 token 列表”
    token_lists: list[list[str]] = [_extract_tokens_from_candidate(c) for c in candidates]

    # 再转成人类可读的 “词1|词2|...” 形式，供大模型判断
    cand_strs = ["|".join(tokens) for tokens in token_lists]

    # 调大模型选择
    result = call_llm_choose_letter(line, cand_strs, model=model)
    if result is None:
        # 兜底：退回候选0
        print("[WARN] LLM 未能给出有效选择，退回候选 A (index=0)")
        return token_lists[0], 0
    
    # === 新增：处理 WARNING 响应 ===
    if result.startswith("WARNING|"):
        # 解析LLM自定义的分词结果
        custom_seg_str = result[8:]  # 去掉 "WARNING|"
        custom_tokens = [token for token in custom_seg_str.split("|") if token]
        if custom_tokens:
            print(f"[INFO] LLM 提供了自定义分词: {'|'.join(custom_tokens)}")
            # 返回自定义分词，并用 -1 表示这不是原候选中的任何一个
            return custom_tokens, -1
        else:
            print("[WARN] LLM 的WARNING响应中未解析出有效分词，退回候选 A")
            return token_lists[0], 0
    
    # 调大模型选字母
    letter = result
    idx = LETTER_LIST.index(letter)
    if idx >= len(token_lists):
        print(f"[WARN] LLM 选择了超出候选范围的字母 {letter}，退回候选 A (index=0)")
        return token_lists[0], 0

    return token_lists[idx], idx

    # 调大模型选字母
    letter = call_llm_choose_letter(line, cand_strs, model=model)

    if letter is None:
        # 兜底：退回候选0（一般是你 DP 打分的最优方案）
        print("[WARN] LLM 未能给出有效选择，退回候选 A (index=0)")
        return candidates[0], 0

    idx = LETTER_LIST.index(letter)
    if idx >= len(candidates):
        # 如果模型选了一个超出范围的字母（例如只有 A/B/C，结果返回了 D），也退回候选0
        print(f"[WARN] LLM 选择了超出候选范围的字母 {letter}，退回候选 A (index=0)")
        return candidates[0], 0

    return candidates[idx], idx
