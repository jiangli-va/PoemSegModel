import jieba

class JiebaSegmenter:
    def __init__(self):
        # 结巴分词默认即可用，如有特定古典诗词词典也可通过 jieba.load_userdict() 加载
        pass
        
    def segment(self, text):
        if not text:
            return ""
        # jieba.cut 返回一个可迭代的生成器，转换为list后用“|”拼起来
        words = list(jieba.cut(text))
        return "|".join(words)
