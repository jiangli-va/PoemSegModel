# run_llm_segmentation.py
# -*- coding: utf-8 -*-
"""
命令行脚本：批量用「Trie+音步候选 + 大模型重排」进行古诗分词。

使用方法（示例）：

  1. 准备 input.txt，每行一条诗句，例如：
       国破山河在
       城春草木深
       空山新雨后
       天气晚来秋

  2. 设置环境变量：
       export OPENAI_API_KEY="你的key"  # windows用 set OPENAI_API_KEY=你的key
       # 如用 DeepSeek：
       # export OPENAI_API_BASE="https://api.deepseek.com"
       # export LLM_MODEL="deepseek-chat"
       # 综合命令：$env:OPENAI_API_KEY="sk-***********"; $env:LLM_MODEL="deepseek-chat"; $env:OPENAI_API_BASE="https://yeysai.com/v1"; python run_llm_segmentation.py --input input.txt --output seg_llm.csv

  3. 运行：
       python run_llm_segmentation.py --input input.txt --output seg_llm.csv

输出 CSV 列包括：
  - line: 原句
  - cand_A / cand_B / cand_C / cand_D / cand_E: 若干候选分词（不足 5 个则为空）
  - llm_choice: LLM 最终选择（A/B/C/D/E 或 'A' 默认）
  - final_seg: 最终采用的分词（用 | 分隔）
"""

import argparse
import csv
import os
from typing import List

from config import load_vocab_and_scores
from candidate_segmenter import (
    generate_candidates_for_line,
    format_tokens,
)
from llm_rerank import choose_best_segmentation


def read_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.strip():
                lines.append(line.strip())
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文本文件路径，每行一条诗句")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--top_k", type=int, default=5, help="每行最多候选数量（<=5 比较适合 LLM）")
    parser.add_argument("--max_word_len", type=int, default=4, help="最大词长")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    top_k = max(1, min(args.top_k, 5))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    print("加载词表与词得分...")
    vocab, word_score = load_vocab_and_scores(min_length=1, max_length=args.max_word_len)
    print(f"词表大小：{len(vocab)}")

    lines = read_lines(input_path)
    print(f"待处理诗句数量：{len(lines)}")

    fieldnames = ["line", "cand_A", "cand_B", "cand_C", "cand_D","cand_E","llm_choice", "final_seg"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        letters = "ABCDE"  # 最多导出前 5 个候选

        for idx, line in enumerate(lines):
            print(f"[{idx+1}/{len(lines)}] 处理：{line}")

            candidates = generate_candidates_for_line(
                line=line,
                vocab=vocab,
                word_score=word_score,
                max_word_len=args.max_word_len,
                top_k=top_k,
            )

            tokens_list: List[List[str]] = [toks for toks, _ in candidates]

            # 默认值（万一候选为空）
            final_tokens: List[str] = []
            llm_choice_letter = ""

            if not candidates:
                print("  [警告] 未生成任何候选分词。")
            elif len(candidates) == 1:
                # 只有一个候选，直接用它
                final_tokens = candidates[0][0]
                llm_choice_letter = "A_DP"
            else:
                # 有多个候选，一律交给 LLM（使用“语法/句法优先”的分词原则）
                best_tokens, best_index = choose_best_segmentation(line, candidates)
                final_tokens = best_tokens
                if best_index == -1:
                    # LLM 提供了自定义分词
                    llm_choice_letter = "WARNING"
                elif best_index >= 0:
                    # 正常选择了某个候选
                    llm_choice_letter = chr(ord("A") + best_index)
                else:
                    # 其他意外情况，兜底
                    llm_choice_letter = "A_DP"
                    print(f"   [注意] 异常 best_index 值: {best_index}")



            row = {
                "line": line,
                "final_seg": format_tokens(final_tokens),
                "llm_choice": llm_choice_letter,
            }

            # 导出前 5 个候选
            for i in range(5):
                key = f"cand_{letters[i]}"
                if i < len(tokens_list):
                    row[key] = format_tokens(tokens_list[i])
                else:
                    row[key] = ""

            writer.writerow(row)
            print(f"[{idx+1}/{len(lines)}] 处理：{line} - 完成")


if __name__ == "__main__":
    main()
