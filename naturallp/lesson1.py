#coding:utf-8
import numpy as np
import pickle
import os
import jieba
import jieba.analyse
import jieba.posseg


# words = u'人们 常 说 生活 是 一 部 教科书 ， 而 血 与 火 的 战争 > 更 是 不可多得 的 教科书'
# 切词
def f1():
    jieba.analyse.set_stop_words('data/stopwords.txt')
    jieba.load_userdict('data/userdict.txt')
    sents = '如果把这句话放在词典中将出错,我要去中国自然博物馆'
    jieba.suggest_freq(('中','将'),True)
    print 'Precise mode:' + '/'.join(jieba.cut(sents))
    print 'All mode:' + '/'.join(jieba.cut(sents, cut_all=True))
    print('cut_for_search:' +  ' '.join(jieba.cut_for_search(sents)))

# 抽取关键词,停用词，自定义字典
def f2():
    jieba.analyse.set_stop_words('data/stopwords.txt')
    # jieba.load_userdict('data/userdict.txt')
    lines = open(r'F:\BaiduYunDownload\Lecture_1\Lecture_1\NBA.txt').read()
    print(' '.join(jieba.analyse.extract_tags(lines, withWeight=False, allowPOS=('v'))))
    print('---' * 30)
    print('/'.join(jieba.analyse.textrank(lines, withWeight=False)))

# 添加词性
def f3():
    words = jieba.posseg.cut('我爱自然语言处理')
    for word,flag in words:
        print('%s %s'%(word,flag))


if __name__ == "__main__":
    f1()