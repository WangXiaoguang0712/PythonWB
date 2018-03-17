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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from twi.anns.ann_bp_multi import ANN_Multi
from matplotlib import pyplot as plt
__author__ = 'T'

"""
参考：
http://blog.csdn.net/eastmount/article/details/50323063
https://www.jianshu.com/p/915b0ab166e5
"""
def segment(text):
    pat_str = u'[‘’“”，、。！？：；%——…《》【】（）\[\]\d]+|(nbsp;)+|(&)+'
    pat = re.compile(pat_str)
    text = re.sub(pat_str, ' ', text)
    outstr = ''
    # 读取停用词
    with open('data/stopwords.txt', 'r') as fr:
        l_stopwords = [line.strip() for line in fr.readlines()]
    for w in jieba.cut(text):
        if w.encode('utf-8') not in l_stopwords:
            outstr += w + ' '
    return outstr.rstrip(' ')

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
                            fw.write(segment(line.strip().strip('\t').strip('\n').strip('\r') + '\r\n'))
                        else:
                            if len(line) > 0:
                                fw.write(segment(line.strip().strip('\t').strip('\n').strip('\r')))


class TextClassify:
    def __init__(self, tsize=0.8, f_size = 50, mtype='NB'):
        self.tsize = tsize
        self.feature_size = f_size
        self.datapath = 'data/classify/'
        self.class_label = self.load_class()
        self.vocabulary = list([])
        self.model = None
        self.mtype = mtype
        self.idf = 0

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

    def prepare_for_fit(self, data, target, is_fit=True, is_idf=True, is_onehot=True):
        l_data_seg = []
        l_data_tar = []
        # 分词并对结果编码
        for d,t in zip(data,target):
            l_data_seg.append(d)
            l_data_tar.append(self.class_label.index(t))

        if not is_fit:
            victorize = CountVectorizer(max_features=self.feature_size, preprocessor=self._remove_noise,
                                        vocabulary=self.vocabulary)
            X = victorize.fit_transform(l_data_seg)
            if is_idf:
                # 手动计算TF-IDF值
                X = X.toarray() * self.idf
            else:
                X = X.toarray()
        else:
            victorize = CountVectorizer(max_features=self.feature_size, preprocessor=self._remove_noise)
            X = victorize.fit_transform(l_data_seg)
            # print(X)
            self.vocabulary = victorize.vocabulary_
            if is_idf:
                transformer = TfidfTransformer()
                X = transformer.fit_transform(X)
                self.idf = transformer.idf_
                X = X.toarray()
            else:
                X = X.toarray()
        if is_onehot:
            # 分类结果 one - hot 编码
            enc = preprocessing.OneHotEncoder()
            enc.fit([[x] for x in range(len(self.class_label))])
            y = enc.transform(np.array(l_data_tar).reshape(-1, 1)).toarray()
        else:
            y = l_data_tar
        return X, y

    def cv(self, data_train, target_train):
        X, y = self.prepare_for_fit(data_train, target_train, is_fit=True, is_idf=True, is_onehot=False)
        result = {}
        def test_model(clf):
            cv = KFold(n_splits=8, shuffle=True, random_state=45)
            scores = cross_val_score(clf, X, y, cv=cv)
            return scores.mean()
        #SVM
        clf = svm.SVC(decision_function_shape='ovr')
        result['SVM'] = test_model(clf)
        #NB
        clf = MultinomialNB()
        result['NB'] = test_model(clf)
        #DT
        clf =  DecisionTreeClassifier(criterion='gini', max_depth=10)
        result['DT'] = test_model(clf)
        #LR
        clf =   LogisticRegression(multi_class='ovr')
        result['LR'] = test_model(clf)
        #KNN
        clf =  KNeighborsClassifier(n_neighbors=5, algorithm='auto')
        result['KNN'] = test_model(clf)

        print(result)
        plt.figure()
        plt.bar(list(result.iterkeys()),list(result.itervalues()))
        plt.show()

    def tuning(self, data_train, target_train):
        X, y = self.prepare_for_fit(data_train, target_train, is_fit=True, is_idf=True, is_onehot=False)
        cv = KFold(n_splits=5, shuffle=True, random_state=45)
        parameters = {'max_depth': [6, 7, 8, 9], 'criterion': ['gini', 'entropy']}
        clf = DecisionTreeClassifier(criterion='gini', max_depth=10)
        grid_obj = GridSearchCV(clf, param_grid=parameters, cv=cv)
        grid_fit = grid_obj.fit(X, y)
        best_model = grid_fit.best_estimator_
        print(grid_fit.grid_scores_)
        print(grid_fit.best_params_)
        print(best_model.score(X, y))


    def fit(self, data_train, target_train):
        # 训练
        if self.mtype == 'NN':
            #  整理数据，方便调用sklearn 方法
            X, y = self.prepare_for_fit(data_train, target_train, is_fit=True)
            self.model = ANN_Multi([self.feature_size, 45, 9])
            self.model.fit(X, y, ANN_Multi.active_simoid)
        elif self.mtype == 'NB':
            X, y = self.prepare_for_fit(data_train, target_train, is_fit=True, is_idf=False, is_onehot=False)
            self.model = MultinomialNB()
            self.model.fit(X, y)
        elif self.mtype == 'DT':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=True, is_idf=True, is_onehot=False)
            self.model = DecisionTreeClassifier(criterion='gini', max_depth=10)
            self.model.fit(X, y)
        elif self.mtype == 'SVM':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=True, is_idf=True, is_onehot=False)
            self.model = svm.SVC(decision_function_shape='ovr')
            self.model.fit(X, y)
        elif self.mtype == 'LR':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=True, is_idf=True, is_onehot=False)
            self.model = LogisticRegression(multi_class='ovr')
            self.model.fit(X, y)
        elif self.mtype == 'KNN':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=True, is_idf=True, is_onehot=False)
            self.model = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
            self.model.fit(X, y)
        else:
            pass

    def predict(self, data_test, target_test):
        if self.mtype == 'NN':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False)
            res_y = self.model.predict(X)
            tmp = [ 1 if np.argmax(y[i]) == res_y[i] else 0 for i in xrange(y.shape[0])]
            print(res_y.reshape(1, -1))
            print(np.argmax(y, axis=1))
        elif self.mtype == 'NB':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=False, is_onehot=False)
            res_y = self.model.predict(X)
            print(res_y)
        elif self.mtype == 'DT':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            res_y = self.model.predict(X)
            print(res_y)
        elif self.mtype == 'SVM':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            res_y = self.model.predict(X)
            print(res_y)
        elif self.mtype == 'LR':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            res_y = self.model.predict(X)
            print(res_y)
        elif self.mtype == 'KNN':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            res_y = self.model.predict(X)
            print(res_y)
        else:
            pass


    def score(self, data_test, target_test):
        if self.mtype == 'NN':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False)
            res_y = self.model.predict(X)
            tmp = [ 1 if np.argmax(y[i]) == res_y[i] else 0 for i in xrange(y.shape[0])]
            print(sum(tmp) * 100.0 / len(tmp))
        elif self.mtype == 'NB':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=False, is_onehot=False)
            model_accuracy = self.model.score(X, y)
            print(model_accuracy)
        elif self.mtype == 'DT':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            model_accuracy = self.model.score(X, y)
            print(model_accuracy)
        elif self.mtype == 'SVM':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            model_accuracy = self.model.score(X, y)
            print(model_accuracy)
        elif self.mtype == 'LR':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            model_accuracy = self.model.score(X, y)
            print(model_accuracy)
        elif self.mtype == 'KNN':
            X, y = self.prepare_for_fit(data_test, target_test, is_fit=False, is_idf=True, is_onehot=False)
            model_accuracy = self.model.score(X, y)
            print(model_accuracy)
        else:
            pass

    def test_statistics(self):
        pass

if __name__ == '__main__':
    # integrate_data()  #  整合文本
    tclf = TextClassify(f_size = 200, mtype='KNN')
    # 分 集
    data_train, data_test, target_train, target_test = tclf.split_dataset()
    #tclf.tuning(data_train, target_train)
    tclf.fit(data_train, target_train)
    #tclf.score(data_test, target_test)


