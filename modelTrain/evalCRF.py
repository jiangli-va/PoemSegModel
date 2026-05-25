import joblib
from pathlib import Path

# 文件路径配置
CRF_MODEL_PATH = Path("crfModel/crf_poem_seg.pkl")
TEST_FILE_PATH = Path("test_gold.txt")
PUNCTUATIONS = set("，。！？；：、“”‘’（）《》〈〉—…,.!?;:\"'()[]{}")
SEP = "|"

# --- 核心特征提取函数（与训练保持一致） ---
def char_features(chars, i):
    char = chars[i]
    features = {
        "bias": 1.0, 
        "char": char, 
        "is_punct": char in PUNCTUATIONS,
        "is_digit": char.isdigit(), 
        "is_ascii": char.isascii(),
    }
    
    if i == 0:
        features["BOS"] = True
    else:
        features.update({
            "-1:char": chars[i-1], 
            "-1:char+char": chars[i-1] + char
        })
        
    if i == len(chars) - 1:
        features["EOS"] = True
    else:
        features.update({
            "+1:char": chars[i+1], 
            "char+1:char": char + chars[i+1]
        })
        
    if i >= 2:
        features["-2:char"] = chars[i-2]
    if i + 2 < len(chars):
        features["+2:char"] = chars[i+2]
        
    return features


def sentence_features(text):
    return [char_features(list(text), i) for i in range(len(text))]


# --- BMES 标签解码函数 ---
def bmes_to_segmented(text, labels):
    words = []
    current = ""
    for char, label in zip(text, labels):
        if label in ["S", "B"] and current:
            words.append(current)
            current = ""
        current += char
        if label in ["S", "E"]:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return SEP.join(words)


# --- 外部预测接口 ---
def predict_crf(crf_model, text):
    """去除首尾空白（含标点视需修改），执行 CRF 分词"""
    if not text:
        return ""
    labels = crf_model.predict_single(sentence_features(text))
    return bmes_to_segmented(text, labels)


if __name__ == "__main__":
    if not CRF_MODEL_PATH.exists():
        print(f"找不到 CRF 模型文件：{CRF_MODEL_PATH}，请确认路径或先运行训练。")
        exit(1)
        
    if not TEST_FILE_PATH.exists():
        print(f"找不到测试集文件：{TEST_FILE_PATH}。")
        exit(1)

    print(f"Loading CRF model from {CRF_MODEL_PATH}...")
    crf = joblib.load(CRF_MODEL_PATH)
    
    print(f"\n========= CRF 分词评测 ({TEST_FILE_PATH}) =========")
    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            poem_text = line.strip()
            if not poem_text:
                continue  # 跳过空行
            
            # 使用模型进行预测并打印
            result = predict_crf(crf, poem_text)
            print(f"原句: {poem_text}")
            print(f"CRF : {result}")
            print("-" * 30)

    print("\n评测分词完毕！")