# _*_ coding:utf-8 _*_
import jieba
import jieba.posseg as psg
import CRFPP
import nltk


__author__ = 'T'

def test_jieba():
    str = u'我和朋友一起去北京故宫博物院参观和闲逛。'
    print ','.join(jieba.cut(str))
    print ','.join(jieba.cut(str,cut_all=True))
    print([(x.word,x.flag) for x in psg.cut(str)])

def test_crf():
    print CRFPP.Model_swigregister

def test_nltk():
    nltk.download()

if __name__ == "__main__":
    test_nltk()