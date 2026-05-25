import os

try:
    import zhconv
    def convert_t2s(text): return zhconv.convert(text, 'zh-cn')
    def convert_s2t(text): return zhconv.convert(text, 'zh-tw')
except ImportError:
    try:
        import opencc
        cc_t2s = opencc.OpenCC('t2s')
        cc_s2t = opencc.OpenCC('s2t')
        def convert_t2s(text): return cc_t2s.convert(text)
        def convert_s2t(text): return cc_s2t.convert(text)
    except ImportError:
        print("警告: 若要进行繁简兼容匹配，建议执行 `pip install zhconv`，否则繁简体不兼容。")
        def convert_t2s(text): return text
        def convert_s2t(text): return text

class MMSegmenter:
    def __init__(self, dict_path):
        self.dict_set = set()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self.dict_set.add(word)
                    self.dict_set.add(convert_s2t(word))
                    self.dict_set.add(convert_t2s(word))
        self.max_len = max([len(w) for w in self.dict_set]) if self.dict_set else 0

    def segment(self, text):
        result = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            # 从最大长度尝试向下匹配
            for j in range(min(self.max_len, n - i), 0, -1):
                word = text[i:i+j]
                if j == 1 or word in self.dict_set \
                   or convert_s2t(word) in self.dict_set \
                   or convert_t2s(word) in self.dict_set:
                    result.append(word)
                    i += j
                    matched = True
                    break
            if not matched:  # 只有当所有的都匹配不上时（单字也匹配不上，理论上不会走到这）
                result.append(text[i])
                i += 1
        return "|".join(result)
