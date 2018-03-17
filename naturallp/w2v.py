# _*_ coding:utf-8 _*_
from gensim.models import Word2Vec
import codecs
import chardet
import os
__author__ = 'T'
#http://blog.csdn.net/MebiuW/article/details/52303622
#http://blog.csdn.net/xuyaoqiaoyaoge/article/details/78571700
#https://spaces.ac.cn/archives/4304/comment-page-1

class TextLoader():
    def __init__(self):
        self.path = 'data/classify/'

    def __iter__(self):
        for f in os.listdir(self.path):
            with open(os.path.join(self.path, f), mode='r') as fr:
                for line in fr.readlines():
                    line = line.decode('utf-8').encode('gbk', 'ignore').decode('gbk').split(' ')
                    yield line

def mode2txt():
    model = Word2Vec(TextLoader(), min_count=1, iter=100)
    model.wv.save_word2vec_format("data/w2v_model.txt", binary=False)  # 保存模型

def txt2list(fp):
    lst = []
    with open(fp, mode='r') as fr:
        for line in fr.readlines():
            line = line.decode('utf-8').encode('gbk', 'ignore').decode('gbk')
            lst.append(line.strip().split())
    return lst

def train_model():
    model = Word2Vec(TextLoader(), min_count=1, iter=100)
    model.save("data/w2v_model")  # 保存模型

def test_model(tag):
    model = Word2Vec.load('data/w2v_model')
    if tag == 1:
        print(model[u'中国'])  # 打印中国的词向量
    elif tag == 2:
        print(model.similarity(u'中国', u'政府'))  # 打印相似度
    elif tag == 3:
        for key, val in model.most_similar(u'志愿'):  # 近义词
            print(key.encode('gbk'))
    elif tag == 4:  # 带修饰的词
        for key, val in model.most_similar(positive=[u'大会',u'召开'], negative=[u'村民']):
            print(key.encode('gbk'))
    elif tag == 5:  # 打印不匹配的词
        print(model.doesnt_match(u"大会 召开 村民 法院".split()))

if __name__ == "__main__":
    # train_model()
    test_model(3)