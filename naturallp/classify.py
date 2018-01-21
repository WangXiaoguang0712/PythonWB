# _*_ coding:utf-8 _*_
import os
import codecs
import jieba
import chardet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

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


class TextClassify():
    def __init__(self, tsize=0.8):
        self.tsize = tsize
        self.datapath = 'data/classify/'

    @staticmethod
    def _segment(text):
        return ' '.join(jieba.cut(text))

    def split_dataset(self):
        # 分 集
        data_train, data_test= [], []
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

    def prepare_for_tfidf(self, data, target):
        data_dict = {}
        # 按分类以字典形式存储
        for idx,item in enumerate(data):
            cls = target[idx]
            if data_dict.has_key(cls):
                data_dict[cls] += ' ' + self._segment(item)
            else:
                data_dict[cls] = self._segment(item)
        # 转换成 数组
        l_cls_words = []
        l_cls_name =  []
        for x in data_dict.items():
            l_cls_words.append(x[1])
            l_cls_name.append(x[0])
        return l_cls_words, l_cls_name

    def fit(self):
        # 分 集
        data_train, data_test, target_train, target_test = self.split_dataset()
        #  整理数据，方便调用sklearn 方法
        l_cls_words, l_cls_name = self.prepare_for_tfidf(data_train, target_train)
        # 转换成文档矩阵
        victorize = CountVectorizer()
        X = victorize.fit_transform(l_cls_words)  # 学习词语词典并返回文档矩阵，矩阵中元素为词语出现的次数
        words = victorize.get_feature_names()  # 获取词袋中所有文本关键词
        # 统计每个词语的TF-IDF值
        tansformer = TfidfTransformer()
        tfidf = tansformer.fit_transform(X)
        train_X = tfidf.toarray()

        # 训练
        lrmode = LogisticRegression()
        # lrmode.fit(train_X,)



    def predict(self):
        pass

    def test_statistics(self):
        pass



if __name__ == '__main__':
    # integrate_data()  #  整合文本
    tclf = TextClassify()
    tclf.fit()


