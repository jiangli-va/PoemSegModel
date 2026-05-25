# trainCFR.py
# Conditional Random Fields (CRF) 条件随机场
# CFR 诗歌分词模型训练

import io
import sys
import re
import matplotlib.pyplot as plt
import json
from pathlib import Path
import joblib
import sklearn_crfsuite
from tqdm import tqdm
#from opencc import OpenCC
#cc_t2s = OpenCC("t2s")

TRAIN_FILE = Path("data/train.jsonl")
VAL_FILE = Path("data/val.jsonl")
TEST_FILE = Path("data/test.jsonl")
OUTPUT_MODEL = Path("outputs/crf_poem_seg.pkl")
SEP = "|"
PUNCTUATIONS = set("，。！？；：、“”‘’（）《》〈〉—…,.!?;:\"'()[]{}")

# 不进行繁简修复
#def normalize_script(text: str) -> str:
#    """将文本转换为简体，去除空格。"""
#    if cc_t2s is None:
#        return text
#    return cc_t2s.convert(text)


def load_jsonl(path: Path):
    """从 jsonl 文件加载样本，返回 (text, segmented) 的列表。"""
    samples = []
    
    strip_chars = " \t\n\r" + "".join(PUNCTUATIONS)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["input"].strip(strip_chars)
            segmented = obj["output"].strip(strip_chars)

            if text and segmented:
                samples.append((text, segmented))

    return samples


def split_segmented(segmented: str):
    """将分词结果按照分隔符切分成词列表，去除空词"""
    return [w for w in segmented.strip().split(SEP) if w]


def map_segmented_to_raw_words(raw_text: str, segmented: str):
    """
    只根据 segmented 的词长序列，在 raw_text 上恢复词序列。
    不比较繁简，不比较字符内容。
    只要求 segmented 去掉分隔符后的字符数等于 raw_text 字符数。

    例：
    raw_text:  對月數諸峰
    segmented: 对|月|数|诸峰
    返回：["對", "月", "數", "諸峰"]
    """
    raw_text = raw_text.strip()
    seg_words = split_segmented(segmented)

    if not raw_text or not seg_words:
        return None

    seg_plain = "".join(seg_words)

    #if len(seg_plain) != len(raw_text):
    #    if len(seg_plain) != len(raw_text) - 1:
    #        return None

    raw_words = []
    cursor = 0

    for seg_word in seg_words:
        word_len = len(seg_word)
        raw_words.append(raw_text[cursor: cursor + word_len])
        cursor += word_len

    if cursor != len(raw_text):
        return None

    return raw_words


def words_to_bmes(words):
    """将词列表转换为 BMES 标签列表。"""
    labels = []

    for word in words:
        if len(word) == 1:
            labels.append("S")  # 单字成词，标记为 S
        elif len(word) == 2:
            labels.extend(["B", "E"])  # 双字成词，标记为 B 和 E
        else:
            labels.append("B")  # 多字成词，首字标记为 B
            labels.extend(["M"] * (len(word) - 2)) # 中间字标记为 M
            labels.append("E") # 末字标记为 E

    return labels


def sample_to_bmes(text: str, segmented: str):
    """将一个样本的文本转换为 BMES 标签列表。"""
    raw_words = map_segmented_to_raw_words(text, segmented)

    if raw_words is None:
        return None

    labels = words_to_bmes(raw_words)

    if len(labels) != len(text):
        return None

    return labels


def char_features(chars, i):
    """提取第 i 个字符的特征，包括当前字符、前后字符、是否标点等。"""
    char = chars[i]

    features = {
        "bias": 1.0,  # 常数特征，帮助模型学习一个全局偏置
        "char": char,  # 当前字符
        "is_punct": char in PUNCTUATIONS,  # 是否是标点符号
        "is_digit": char.isdigit(),  # 是否是数字
        "is_ascii": char.isascii(),  # 是否是 ASCII 字符
    }

    if i == 0:
        features["BOS"] = True  # 句首特征
    else:
        prev_char = chars[i - 1] # 前一个字符
        features.update({
            "-1:char": prev_char,
            "-1:is_punct": prev_char in PUNCTUATIONS,
            "-1:char+char": prev_char + char,
        })

    if i == len(chars) - 1:
        features["EOS"] = True  # 句尾特征
    else:
        next_char = chars[i + 1]
        features.update({
            "+1:char": next_char,
            "+1:is_punct": next_char in PUNCTUATIONS,
            "char+1:char": char + next_char,
        })

    if i >= 2:
        features["-2:char"] = chars[i - 2]
        features["-2:-1:char"] = chars[i - 2] + chars[i - 1]

    if i + 2 < len(chars):
        features["+2:char"] = chars[i + 2]
        features["+1:+2:char"] = chars[i + 1] + chars[i + 2]

    return features


def sentence_features(text):
    chars = list(text)
    return [char_features(chars, i) for i in range(len(chars))]


def build_dataset(samples, name="dataset"):
    """将样本列表转换为特征和标签的列表，进行 BMES 标签化，并输出异常信息。"""
    X = []
    y = []

    skipped = 0
    length_mapped = 0

    for text, segmented in tqdm(samples, desc=f"Building {name}"):
        labels = sample_to_bmes(text, segmented)

        if labels is None:
            skipped += 1
            continue

        if "".join(split_segmented(segmented)) != text:
            length_mapped += 1

        X.append(sentence_features(text))
        y.append(labels)

    print(f"{name}: valid={len(X)}, skipped={skipped}, length_mapped={length_mapped}") 
    return X, y


def bmes_to_segmented(text, labels):
    """将 BMES 标签列表转换回分词结果字符串。"""
    words = []
    current = ""

    for char, label in zip(text, labels):
        if label == "S":
            if current:
                words.append(current)
                current = ""
            words.append(char)

        elif label == "B":
            if current:
                words.append(current)
            current = char

        elif label == "M":
            if current:
                current += char
            else:
                current = char

        elif label == "E":
            if current:
                current += char
                words.append(current)
                current = ""
            else:
                words.append(char)

        else:
            if current:
                words.append(current)
                current = ""
            words.append(char)

    if current:
        words.append(current)

    return SEP.join(words)


def words_to_spans(words):
    """将词列表转换为文本中的位置跨度集合，跨度表示为 (start, end) 的元组，其中 end 是开区间。"""
    spans = set()
    start = 0

    for word in words:
        end = start + len(word)
        spans.add((start, end))
        start = end

    return spans


def segmented_to_spans_on_raw(text, segmented):
    """将分词结果映射回原文本的词位置跨度集合。"""
    raw_words = map_segmented_to_raw_words(text, segmented)

    if raw_words is None:
        return None

    return words_to_spans(raw_words)


def evaluate(crf, samples, name="test"):
    """评估 CRF 模型在给定样本上的性能，计算精确率、召回率、F1 分数和完全匹配率，并输出错误示例。"""
    total_pred = 0
    total_gold = 0
    total_correct = 0
    exact_match = 0
    valid = 0
    skipped = 0

    examples = []

    for text, gold_segmented in tqdm(samples, desc=f"Evaluating {name}"):
        gold_spans = segmented_to_spans_on_raw(text, gold_segmented)

        if gold_spans is None:
            skipped += 1
            continue

        pred_labels = crf.predict_single(sentence_features(text))
        pred_segmented = bmes_to_segmented(text, pred_labels)
        pred_words = split_segmented(pred_segmented)
        pred_spans = words_to_spans(pred_words)

        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
        total_correct += len(pred_spans & gold_spans)

        gold_words = map_segmented_to_raw_words(text, gold_segmented)
        gold_repaired = SEP.join(gold_words)

        if pred_segmented == gold_repaired:
            exact_match += 1

        if len(examples) < 10 and pred_segmented != gold_repaired:
            examples.append((text, gold_repaired, pred_segmented))

        valid += 1

    accuracy = total_correct / valid if valid else 0.0
    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = exact_match / valid if valid else 0.0

    result = {
        "valid": valid,
        "skipped": skipped,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
    }

    print(f"\n{name} result:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print(f"\n{name} error examples:")
    for text, gold, pred in examples:
        print("-" * 60)
        print("text:", text)
        print("gold:", gold)
        print("pred:", pred)

    return result


def train_crf(X_train, y_train):
    """训练 CRF 模型，使用 L-BFGS 算法，并捕获日志绘制 loss 曲线。"""
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",  
        c1=0.1,  
        c2=0.1,  
        max_iterations=200,
        all_possible_transitions=True, 
        verbose=True,
    )

    print("捕获训练日志以绘制损失曲线...")
    # 拦截标准错误/标准输出（因操作系统和库版本不同，CRF日志可能输出在 stderr 或 stdout）
    old_stdout = sys.stdout 
    old_stderr = sys.stderr
    log_capture = io.StringIO()
    sys.stdout = log_capture
    sys.stderr = log_capture

    try:
        crf.fit(X_train, y_train)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # 获取捕获的日志字符串并打印在屏幕上
    log_text = log_capture.getvalue()
    print(log_text)

    # 用正则抓取 L-BFGS 的 loss。日志格式类似: Iter 1   time=0.02  loss=6615.59  active=...
    losses = []
    for match in re.finditer(r"loss=([0-9.]+)", log_text):
        losses.append(float(match.group(1)))

    if losses:
        # 画出 loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', markersize=3, label="L-BFGS Loss")
        plt.title("CRF Training Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        
        # 将图片保存下来
        plot_path = Path("outputs/loss_curve.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Loss 曲线已保存到: {plot_path}")
    else:
        print("未在日志中匹配到 loss 信息，无法绘制曲线。")

    return crf


def main():
    print("Loading data...")
    train_samples = load_jsonl(TRAIN_FILE)
    val_samples = load_jsonl(VAL_FILE)
    test_samples = load_jsonl(TEST_FILE)

    print(f"train samples: {len(train_samples)}")
    print(f"val samples: {len(val_samples)}")
    print(f"test samples: {len(test_samples)}")

    X_train, y_train = build_dataset(train_samples, "train")
    X_val, y_val = build_dataset(val_samples, "val")

    print("Training CRF...")
    crf = train_crf(X_train, y_train)

    evaluate(crf, val_samples, "val")
    evaluate(crf, test_samples, "test")

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, OUTPUT_MODEL)

    print(f"Model saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()