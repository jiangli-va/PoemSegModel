import re
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class DeepSeekSegmenter:
    def __init__(self, api_key):
        """
        初始化 DeepSeek 接口
        """
        self.api_key = api_key
        if OpenAI and self.api_key and self.api_key != "YOUR_DEEPSEEK_API_KEY":
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        else:
            self.client = None
            
    def segment(self, text):
        if not text:
            return ""
        if not self.client:
            return "Please provide valid DeepSeek API KEY and install openai"
            
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
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 或者 deepseek-coder
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.0
            )
            ans = response.choices[0].message.content
            # 去除可能存在的 <think> 标签
            ans_clean = re.sub(r"<think>.*?</think>\s*", "", ans, flags=re.DOTALL)
            return ans_clean.strip()
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return str(e)
