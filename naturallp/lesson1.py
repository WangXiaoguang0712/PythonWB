#coding:utf-8
import numpy as np
import pickle
import os
import chardet
import jieba
import jieba.analyse
import jieba.posseg


# words = u'人们 常 说 生活 是 一 部 教科书 ， 而 血 与 火 的 战争 > 更 是 不可多得 的 教科书'
# 切词
def f1():
    jieba.analyse.set_stop_words('data/stopwords.txt')
    sents = '小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
    print('cut Precise mode:' + '/'.join(jieba.cut(sents)))
    print('cut All mode:' + '/'.join(jieba.cut(sents, cut_all=True)))
    print('cut_for_search:' +  '/'.join(jieba.cut_for_search(sents)))
    print('-----' * 40)
    sents = '如果把这句话放在词典中将出错'
    print('Normal:' + '/'.join(jieba.cut(sents)))
    jieba.suggest_freq(('中','将'),True)
    print('cut suggest_freq:' + '/'.join(jieba.cut(sents)))
    print('-----' * 40)
    jieba.load_userdict('data/userdict.txt')
    print('cut load_userdict:' + '/'.join(jieba.cut(sents)))
    print('-----' * 40)
    jieba.del_word('出错')
    print('cut del_word:' + '/'.join(jieba.cut(sents)))
    print('-----' * 40)
    print('extract_tags del_word:' + '/'.join(jieba.analyse.extract_tags(sents, topK=20)))

# 添加词性
def f3():
    words = jieba.posseg.cut('我爱自然语言处理')
    for word,flag in words:
        print('%s %s'%(word,flag))


# 返回词语在原文的起止位置
def f5():
    result = jieba.tokenize(u'永和服装饰品有限公司')
    for tk in result:
        print("word %s\t\t start:%s\t\t end:%s" %(tk[0], tk[1], tk[2]))

def stopwords(file_path):
    with open(file_path, mode= 'r', encoding='utf-8') as fr:
        return [line.strip() for line in fr.readlines()]

# 停用词
def f6():
    outstr = ''
    sents = u'小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
    l_stopwords = stopwords('data/stopwords.txt')
    print(l_stopwords[2])
    for w in jieba.cut(sents):
        if w.encode('utf-8') not in l_stopwords:
            outstr += w + ' '
    return outstr.rstrip(' ')

if __name__ == "__main__":
    res = f6()
    print(res)