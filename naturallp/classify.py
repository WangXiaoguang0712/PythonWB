# _*_ coding:utf-8 _*_
import os
import re
import codecs
import jieba
import chardet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from twi.anns.ann_bp_multi import ANN_Multi

__author__ = 'T'

"""
参考：
http://blog.csdn.net/eastmount/article/details/50323063
"""

def integrate_data():
    path_sour = 'C:\Users\T\Downloads\Lecture_2\Naive-Bayes-Text-Classifier\Database\SogouC\Sample'
    corpus = 'data/classify/'
    folder_list = os.listdir(path_sour)
    for folder in folder_list:
        subfolder = os.path.join(path_sour, folder)
        with codecs.open(os.path.join(corpus, folder + '.txt'), 'a', 'utf-8') as fw:
            for fn in os.listdir(subfolder):
                f = os.path.join(subfolder, fn)
                with codecs.open(f, 'r', 'utf-8') as fr:
                    lines = fr.readlines()
                    for line in lines:
                        if line == lines[-1]:
                            fw.write(line.strip().strip('\t').strip('\n').strip('\r') + '\r\n')
                        else:
                            fw.write(line.strip().strip('\t').strip('\n').strip('\r'))


class TextClassify:
    def __init__(self, tsize=0.8, f_size = 50):
        self.tsize = tsize
        self.feature_size = f_size
        self.datapath = 'data/classify/'
        self.class_label = self.load_class()
        print([[x] for x in range(len(self.class_label))])

    def load_class(self):
        l = []
        with open(r'data\classify-class.txt' ,'r') as fr:
            for line in fr.readlines():
                l.append(line.split('\t')[0])
        return l

    @staticmethod
    def _segment(text):
        outstr = ''
        # 读取停用词
        with open('data/stopwords.txt', 'r') as fr:
            l_stopwords = [line.strip() for line in fr.readlines()]
        for w in jieba.cut(text):
            if w.encode('utf-8') not in l_stopwords:
                outstr += w + ' '
        return outstr.rstrip(' ')

    def split_dataset(self):
        # 分 集
        data_train, data_test = [], []
        target_train, target_test = [], []
        for f in os.listdir(self.datapath):
            with codecs.open(os.path.join(self.datapath,f), 'r', 'utf-8') as fo:
                lines = fo.readlines()
                class_name = [f[:f.index('.')]] * len(lines)
                x_train, x_test, y_train, y_test = train_test_split(lines , class_name, test_size=0.2)
                data_train.extend(x_train)
                data_test.extend(x_test)
                target_train.extend(y_train)
                target_test.extend(y_test)
        return data_train, data_test, target_train, target_test

    def _remove_noise(self, doc):
        noise_pattern = r'\d+\.?\d+|[a-z]+|[A-Z]+'
        clean_doct = re.sub(noise_pattern, '', doc)
        return clean_doct

    def prepare_for_tfidf(self, data, target):
        l_data_seg = []
        l_data_tar = []

        # 分词并对结果编码
        for d,t in zip(data,target):
            l_data_seg.append(self._segment(d))
            l_data_tar.append(self.class_label.index(t))

        # 转换成文档矩阵 计算tfidf
        victorize = CountVectorizer(max_features=self.feature_size, preprocessor=self._remove_noise)
        X = victorize.fit_transform(l_data_seg)  # 学习词语词典并返回文档矩阵，矩阵中元素为词语出现的次数
        words = victorize.get_feature_names()  # 获取词袋中所有文本关键词
        #for item in words:
        #    print(item)
        # 统计每个词语的TF-IDF值
        tansformer = TfidfTransformer()
        tfidf = tansformer.fit_transform(X)
        X = tfidf.toarray()

        # 分类结果 one - hot 编码
        enc = preprocessing.OneHotEncoder()
        enc.fit([[x] for x in range(len(self.class_label))])
        y = enc.transform(np.array(l_data_tar).reshape(-1, 1)).toarray()

        return X, y

    def fit(self):
        # 分 集
        data_train, data_test, target_train, target_test = self.split_dataset()
        #  整理数据，方便调用sklearn 方法
        X, y = self.prepare_for_tfidf(data_train, target_train)
        print(X[:, :self.feature_size].shape,y.shape)

        # 训练
        m = ANN_Multi([self.feature_size, 45, 9])
        m.fit(X[:, :self.feature_size], y, ANN_Multi.active_simoid)

        # 测试
        X, y = self.prepare_for_tfidf(data_test, target_test)
        res_y = m.predict(X[:, :self.feature_size])
        tmp = [ 1 if np.argmax(y[i]) == res_y[i] else 0 for i in xrange(y.shape[0])]
        print(res_y)
        print(np.argmax(y, axis=1))
        print(sum(tmp) * 100.0 / len(tmp))




    def test(self):
        pass


    def predict(self):
        pass

    def test_statistics(self):
        pass



if __name__ == '__main__':
    # integrate_data()  #  整合文本
    tclf = TextClassify(f_size = 50)
    tclf.fit()


