# 读取excel表格，构建Trie数

# import pandas as pd
# from cyac import Trie
import pandas as pd

# ========= 兼容处理：没有 cyac 也能正常导入 =========
try:
    from cyac import Trie  # 原来的高性能 Trie
except ModuleNotFoundError:
    # 简单兜底实现：只提供 add 和 __contains__，满足本项目需求
    class Trie:
        def __init__(self):
            self._words = set()

        def add(self, word):
            self._words.add(word)

        def __contains__(self, word):
            return word in self._words
# ========= 兼容处理结束 =========

def read_vocab(vocab_path, min_length, max_length=-1):
    df_vocab = pd.read_excel(vocab_path, sheet_name='Sheet1')
    # print(df_vocab.columns.values)
    '''
    ['词条' '长度' '频度' 'MI' '包含词典数' '古汉大辞典孙' '古今词典孙' '古汉常用字典岂' '古>汉大词典辞海岂' '典故词典岂'
    'bigram胡' '搜韵典故' 'type' '字频1' '字频2' '子语料总频度' '子语料正常频度' '子语料不正>常频度'
    '子语料其他频度' '调整比例' '调整后总频度' '调整标识']
    '''
    # main_data = df_vocab[['词条', '长度', '频度']]
    if max_length <= 0:
        vocab = list(df_vocab['词条'])
        vocab_freq_dict = dict(zip(df_vocab['词条'], df_vocab['频度']))
        return vocab, vocab_freq_dict
    else:
        word_to_seg_of_length = {}
        vocab_of_length = {}
        for i in range(min_length, max_length+1):
            word_to_seg_of_length[f'length_{i}'] =[df_vocab['词条'][idx] for idx in range(len(df_vocab)) if df_vocab['长度'][idx]==i] 
        for i in range(min_length, max_length+1):
            vocab_of_length[f'to_seg_length_{i}'] = [df_vocab['词条'][idx] for idx in range(len(df_vocab)) if df_vocab['长度'][idx]<i]
        return word_to_seg_of_length, vocab_of_length

def get_attr(vocab_path, attr_name, min_length, max_length=-1):
    '''
    attr_names: ['词条' '长度' '频度' 'MI' '包含词典数' '古汉大辞典孙' '古今词典孙' '古汉常用字典岂' '古>汉大词典辞海岂' '典故词典岂'
    'bigram胡' '搜韵典故' 'type' '字频1' '字频2' '子语料总频度' '子语料正常频度' '子语料不正>常频度'
    '子语料其他频度' '调整比例' '调整后总频度' '调整标识']
    '''
    df_vocab = pd.read_excel(vocab_path, sheet_name='Sheet1')
    # print(df_vocab.columns.values)
    if max_length <= 0:
        vocab_attr_dict = dict(zip(df_vocab['词条'], df_vocab[attr_name]))
        return vocab_attr_dict
    else:
        vocab_attr_dict_of_length = {}
        for i in range(min_length, max_length+1):
            vocab_attr_dict_of_length[f'length_{i}'] ={df_vocab['词条'][idx]:df_vocab[attr_name][idx] for idx in range(len(df_vocab)) if df_vocab['长度'][idx]==i}
        return vocab_attr_dict_of_length


def build_trie_from_vocab(vocab):
    trie = Trie()
    for word in vocab:
        trie.insert(word)
    return trie

def build_trie():
    vocab_path = './古汉语词典及词频5以上bigram总结表v9.xlsx'
    vocab, vocab_freq_dict = read_vocab(vocab_path, min_length=-1)
    print(f'Read vocab of length {len(vocab)} from {vocab_path}.')
    trie = build_trie_from_vocab(vocab)
    return vocab, vocab_freq_dict, trie


def build_trie_for_wordseg():
    min_length = 2
    max_length = 3
    vocab_path = 'C:\\Users\\Lenovo\\Desktop\\NLP\\诗歌辅助创作\\分词\\古汉语词典及词频5以上bigram总结表v9.xlsx'
    word_to_seg_of_length, vocab_of_length = read_vocab(vocab_path=vocab_path, min_length=min_length, max_length=max_length)
    trie_of_length = {}
    for i in range(min_length, max_length+1):
        trie_of_length[f'to_seg_length_{i}'] = build_trie_from_vocab(vocab_of_length[f'to_seg_length_{i}'])
    return word_to_seg_of_length, trie_of_length