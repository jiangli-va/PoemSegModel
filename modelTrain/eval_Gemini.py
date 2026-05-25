import re
import time
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class GeminiSegmenter:
    def __init__(self, api_key, base_url):
        """
        初始化 Gemini 接口 (基于 OpenAI 兼容格式)
        """
        self.api_key = api_key
        self.base_url = base_url
        if OpenAI and self.api_key and self.api_key != "YOUR_GEMINI_API_KEY":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    def segment(self, text, max_retries=5):
        if not text:
            return ""
        if not self.client:
            return "Please provide valid Gemini API KEY/URL and install openai"
            
        instruction = (
            "请对下面的古典诗词文本进行中文分词，要求如下。"
            "0. 总体原则：根据句意；应分尽分；实事求是；合并词、分短语。"
            "1. 功能词、虚词应尽量单独成词。"
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
        
        # 添加重试机制解决API不稳定断连、截断问题
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gemini-1.5-pro", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.0
                )
                ans = response.choices[0].message.content
                
                # 万一有额外标签，安全起见过滤
                ans_clean = re.sub(r"<think>.*?</think>\s*", "", ans, flags=re.DOTALL).strip()
                
                # 再次校验：如果输出了奇怪的空值或因未知原因被砍去大半，认为是API异常，触发重试
                if not ans_clean or len(ans_clean.replace("|", "")) < len(text) - 2:
                    if attempt < max_retries - 1:
                        time.sleep(1) # 短暂等待后重发
                        continue
                    else:
                        return ans_clean # 如果已经最后一次了，只能返回残缺内容
                        
                return ans_clean
                
            except Exception as e:
                # 捕获HTTP超时、速率限制异常并重试
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return str(e)
