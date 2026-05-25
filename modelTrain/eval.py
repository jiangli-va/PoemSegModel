import csv
import time
from pathlib import Path

# 导入所有评测类
from eval_MM import MMSegmenter
from eval_BMM import BMMSegmenter
from eval_CRF_wrapper import CRFSegmenter
from eval_LoRA import LoRASegmenter
from eval_jieba import JiebaSegmenter
from eval_QwenBase import QwenBaseSegmenter
from eval_DeepSeek import DeepSeekSegmenter
from eval_Gemini import GeminiSegmenter

def get_words(text_split):
    words = text_split.split('|')
    spans = set()
    idx = 0
    for w in words:
        if w: 
            spans.add((w, idx, idx + len(w)))
            idx += len(w)
    return spans

def compute_metrics(gold_list, pred_list):
    sentence_correct = 0
    total_sentences = len(gold_list)
    
    total_gold_words = 0
    total_correct_words = 0

    for gold_str, pred_str in zip(gold_list, pred_list):
        if gold_str == pred_str:
            sentence_correct += 1
            
        gold_spans = get_words(gold_str)
        pred_spans = get_words(pred_str)
        
        total_gold_words += len(gold_spans)
        total_correct_words += len(gold_spans.intersection(pred_spans))

    sent_acc = sentence_correct / total_sentences if total_sentences > 0 else 0
    word_acc = total_correct_words / total_gold_words if total_gold_words > 0 else 0
    
    return sent_acc, word_acc

def main():
    dict_path = "dict_all.txt"
    test_file = "test.csv"
    output_file = "test_results.csv"
    
    # API 密钥配置 
    DEEPSEEK_API_KEY = "sk-658cb973a1bb423cb01d53b9b1a3246a"
    GEMINI_API_KEY = "sk-4cBSxziUBClI5bcyPKytRZV7gZrypsRjabOLc7FK6e1Dixjn"
    GEMINI_BASE_URL = "https://yeysai.com/v1" 
    
    # 基础模型可根据实际情况修改
    qwen_base_model = "Qwen/Qwen3-8B" 
    lora_path = "Qwen3-8B-poemseg" 
    
    print("正在加载模型，请稍候...")
    
    # 词典及CRF模型
    print("加载 MM 模型...")
    mm_seg = MMSegmenter(dict_path)
    print("加载 BMM 模型...")
    bmm_seg = BMMSegmenter(dict_path)
    print("加载 CRF 模型...")
    crf_seg = CRFSegmenter("crfModel/crf_poem_seg.pkl")
    print("加载 jieba 模型...")
    jieba_seg = JiebaSegmenter()
    
    # API模型
    print("加载 DeepSeek API 模型")
    ds_seg = DeepSeekSegmenter(DEEPSEEK_API_KEY)
    print("加载 Gemini API 模型")
    gemini_seg = GeminiSegmenter(GEMINI_API_KEY, GEMINI_BASE_URL)
    
    # LLM本地模型
    print("加载 Qwen3-8B-LoRA 模型...")
    try:
        lora_seg = LoRASegmenter(qwen_base_model, lora_path)
        use_lora = True
    except Exception as e:
        print(f"LoRA模型加载失败 (" + str(e) + ")，跳过。")
        use_lora = False

    print("加载 Qwen3-8B 原始模型...")
    try:
        base_seg = QwenBaseSegmenter(qwen_base_model)
        use_base = True
    except Exception as e:
        print(f"Qwen3-8B原始模型加载失败 (" + str(e) + ")，跳过。")
        use_base = False

    # 读取测试数据
    data = []
    print(f"读取测试集数据：{test_file}")
    with open(test_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    # 开始预测
    print("开始预测...")
    gold_answers = []
    
    # 每个模型预测结果列表
    preds = {
        'MM': [], 'BMM': [], 'CRF': [], 'jieba': [],
        'DeepSeek': [], 'Gemini': [], 'QwenBase': [], 'LoRA': []
    }
    
    for idx, row in enumerate(data):
        text = row['input']
        gold = row['answer']
        gold_answers.append(gold)
        
        preds['MM'].append(mm_seg.segment(text))
        preds['BMM'].append(bmm_seg.segment(text))
        preds['CRF'].append(crf_seg.segment(text))
        preds['jieba'].append(jieba_seg.segment(text))
        
        # API调用
        preds['DeepSeek'].append(ds_seg.segment(text))
        preds['Gemini'].append(gemini_seg.segment(text))
        
        if use_base:
            preds['QwenBase'].append(base_seg.segment(text))
        else:
            preds['QwenBase'].append("")
            
        if use_lora:
            preds['LoRA'].append(lora_seg.segment(text))
        else:
            preds['LoRA'].append("")
            
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1} / {len(data)} 条数据")

    # 计算指标并打印
    print("\n========== 评测结果 ==========")
    for model_name, pred_list in preds.items():
        if model_name in ['LoRA', 'QwenBase'] and not eval(f"use_{model_name.lower().replace('qwenbase', 'base')}"):
            continue # 跳过未成功加载的本地模型指标
        if model_name in ['DeepSeek', 'Gemini'] and (pred_list and "Please provide valid" in pred_list[0]):
            continue # 跳过未配置API KEY的指标统计
            
        s_acc, w_acc = compute_metrics(gold_answers, pred_list)
        print(f"[{model_name}] 句正确率: {s_acc:.2%} | 词正确率: {w_acc:.2%}")
        
    print(f"\n保存结果至：{output_file}")
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ["category", "input", "answer", "MM_pred", "BMM_pred", "CRF_pred", "jieba_pred", "DeepSeek_pred", "Gemini_pred", "QwenBase_pred", "LoRA_pred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(data):
            out_row = {
                "category": row.get("category", ""),
                "input": row["input"],
                "answer": row["answer"],
                "MM_pred": preds['MM'][idx],
                "BMM_pred": preds['BMM'][idx],
                "CRF_pred": preds['CRF'][idx],
                "jieba_pred": preds['jieba'][idx],
                "DeepSeek_pred": preds['DeepSeek'][idx],
                "Gemini_pred": preds['Gemini'][idx],
                "QwenBase_pred": preds['QwenBase'][idx],
                "LoRA_pred": preds['LoRA'][idx]
            }
            writer.writerow(out_row)
            
    print("评测完成！")

if __name__ == "__main__":
    main()
