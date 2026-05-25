# dataPrepare.py
# 处理训练数据

import json
import csv
import random
from pathlib import Path
from opencc import OpenCC

INPUT_DATA = Path("../Segmentation/seg_llm.csv")
OUTPUT_DIR = Path("./data")

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

random.seed(42)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

INSTRUCTION = (
    "请对下面的古典诗词文本进行中文分词，要求如下。"
    "0. 总体原则：根据句意；应分尽分；实事求是；合并词、分短语。"
    "1. 功能词、虚词（如“或、又、无、兮、其、乃、遂、也、矣”等副词、连词、语气词、否定词）应尽量单独成词。"
    "2. 语义紧密、习惯搭配或专有名词的词组可以合并成多字词"
    "3. 有些地方的分词可能存在多种合理的方案，请你选择最符合句意表达的那一项。"
    "4. 不允许增删、更改任何汉字，只比较不同的切分位置；不可机械追求最长词或最多单字，每个词包含字数一般最好不超过4个。"
    "5. 诗句里面重要的地名、人名、专有名词、固定短语，应当合并成词"
    "要求保持原文字符不增不删，不添加解释，只输出分词结果，词语之间用“|”分隔。"
)

cc_t2s = OpenCC("t2s")  # 繁体转简体

def strip_sep(sgmented: str, sep: str = "|")->str:
    return sgmented.replace(sep, "")

def repair(raw_text:str, segmented_text:str, sep:str = "|") -> str | None:
    raw_text = raw_text.strip()
    segmented_text = segmented_text.strip()

    tokens = [tok for tok in segmented_text.split(sep) if tok]
    segmented_plain = "".join(tokens)

    # 原本就完全一致
    if segmented_plain == raw_text:
        return sep.join(tokens)

    # 繁简归一后等价，且字符长度一致，可以按 token 长度把边界映射回原文
    if cc_t2s.convert(segmented_plain) == cc_t2s.convert(raw_text) and len(segmented_plain) == len(raw_text):
        repaired_tokens = []
        cursor = 0

        for token in tokens:
            token_len = len(token)
            repaired_tokens.append(raw_text[cursor: cursor + token_len])
            cursor += token_len

        if cursor == len(raw_text):
            return sep.join(repaired_tokens)

    # 对不齐的样本不要硬修
    return None

def remove_line(text: str) -> str:
    """去掉分词结果中的竖线，检查字数是否相等。由于很多分词结果将繁体改写为简体，所以不会完全相同"""
    return "".join(text.split("|"))

def build_sample(raw_text:str, segmented_text:str) -> dict:
    return{
        "instruction": INSTRUCTION,
        "input": raw_text,
        "output": segmented_text
    }

def load_data(csv_path: Path) -> list[dict]:
    samples = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print("正在加载数据...")
        for row in reader:
            raw_text = row["line"]
            segmented_text = row["final_seg"]
            if len(remove_line(segmented_text)) != len(raw_text):
                pass
            sample = build_sample(raw_text, segmented_text)
            samples.append(sample)
    print(f"数据加载完成，共 {len(samples)} 条样本。")
    return samples

def save_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"数据已保存到 {path}，共 {len(samples)} 条样本。")

def split_data(samples: list[dict]):
    random.shuffle(samples)
    total = len(samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    print(f"数据划分完成：{len(train_samples)} 训练样本，{len(val_samples)} 验证样本，{len(test_samples)} 测试样本。")
    return train_samples, val_samples, test_samples

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    samples = load_data(INPUT_DATA)
    train_samples, val_samples, test_samples = split_data(samples)

    save_jsonl(train_samples, TRAIN_FILE)
    save_jsonl(val_samples, VAL_FILE)
    save_jsonl(test_samples, TEST_FILE)