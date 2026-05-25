# 基于CRF和LoRA的古诗分词模型训练

## 任务描述

中文分词是中文信息处理中的基础任务。与现代汉语相比，古诗文本具有语句凝练、语义模糊、专名和典故密集等特点，通用的面向现代汉语的分词方法难以稳定处理其词语边界。本项目面向古诗自动分词任务，构建古典诗歌分词训练数据，并分别采用条件随机场（CRF）和基于 Qwen3-8B 的 LoRA 参数高效微调方法训练分词模型。最后对传统分词方法、CRF、LoRA 微调模型、通用分词工具和若干闭源大语言模型进行定量和定性比较，分析不同方法在一般诗句、人名、地名和典故等场景下的表现。

## 文件结构

```
├── Segmentation/      # 训练数据构建
├── modelTrain/        # 模型训练与评估
├── requirements.txt   
├── 作业word文档
└── README.md
```

## 环境配置

```dotnetcli
pip install -r requirements.txt
```

### 参考:

- github链接：https://github.com/jiangli-va/PoemSegModel
- huggingface链接：
  - Qwen3-8B-LoRA: https://huggingface.co/jiangli-va/qwen3-8b-poemseg-lora
  - CRF: https://huggingface.co/jiangli-va/crf-poemseg

请引用：