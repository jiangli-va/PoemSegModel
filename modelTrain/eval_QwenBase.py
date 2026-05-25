import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class QwenBaseSegmenter:
    def __init__(self, base_model_name):
        """
        初始化纯原始的 Qwen 基座模型 (无LoRA权重)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 载入Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # 2. 载入原始模型
        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            **model_kwargs
        )
        self.model.eval()

    def segment(self, text):
        if not text:
            return ""
            
        instruction = (
            "请对下面的古典诗词文本进行中文分词，要求如下。"
            "0. 总体原则：根据句意；应分尽分；实事求是；合并词、分短语。"
            "1. 功能词、虚词（如“或、又、无、兮、其、乃、遂、也、矣”等副词、连词、语气词、否定词）应尽量单独成词。"
            "2. 语义紧密、习惯搭配或专有名词的词组可以合并成多字词"
            "3. 有些地方的分词可能存在多种合理的方案，请你选择最符合句意表达的那一项。"
            "4. 不允许增删、更改任何汉字，只比较不同的切分位置；不可机械追求最长词或最多单字，每个词包含字数一般最好不超过4个。"
            "5. 诗句里面重要的地名、人名、专有名词、固定短语，应当合并成词要求保持原文字符不增不删，不添加解释，只输出分词结果，词语之间用“|”分隔。"
        )
        
        system_prompt = (
            "你是一个古典诗歌中文分词模型。"
            "请严格按照用户要求输出分词结果，不要解释，不要改写原文。"
        )
        
        user_prompt = f"{instruction}\n\n待分词文本：{text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response_clean = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
        return response_clean.strip()
