# _*_ coding:utf-8 _*_

import os
import re
import pyprind
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score

def test_countvector():
    count = CountVectorizer()
    docs = np.array(['The sun is shining',
                     'The weather is sweet',
                     'The sun is shining and the weather is sweet'])
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    print(bag.toarray())
    tfidf = TfidfTransformer()
    np.set_printoptions(2)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
    print(tfidf.idf_)


def process_data():
    pbar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = r'E:\PyData\aclImdb\%s\%s' %(s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), encoding='gb18030', errors='ignore') as infile:
                    txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    df.columns = ['review', 'sentiment']
    df = df.reindex(np.random.permutation(df.index))
    df['review'] = df['review'].apply(preprocessor)
    df.to_csv('data/imdb.csv')


def preprocessor(str):
    text = re.sub('<[^>]*>', '', str)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
    return text


def tokenizer(text):
    return  text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in text.split()]

def optimize(X, y):
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    lr_pipe = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__tokenizer': [tokenizer],
                   'clf__penalty': [ 'l2'],
                   'clf__C': [1.0, 10.0]}]
    gs_lr_tfidf = GridSearchCV(estimator=lr_pipe, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)
    gs_lr_tfidf.fit(X, y)
    print('Best parameter set: %s' % gs_lr_tfidf.best_params_)


def best_model(X, y, X_test, y_test):
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, ngram_range=(1, 1),
                            tokenizer=tokenizer)
    lr = LogisticRegression(random_state=0, penalty='l2', C=10.0)
    lr_pipe = Pipeline([('vect', tfidf), ('clf', lr)])
    lr_pipe.fit(X, y)
    scores = cross_val_score(estimator=lr_pipe, X=X, y=y, cv=5)
    print('train CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    scores = cross_val_score(estimator=lr_pipe, X=X_test, y=y_test, cv=5)
    print('test CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def test_waicun_learning(path):
    stop = stopwords.words('english')
    print(stop)
    def tokennizer(str):
        text = re.sub('<[^>]*>', '', str)
        emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
        tokenized = [ w for w in text.split() if w not in stop]
        return tokenized

    def stream_docs(path):
        with open(path, encoding='gb18030', errors='ignore') as f:
            next(f)
            for line in f:
                text, label = line[:-3], int(line[-2])
                yield text, label

    def get_minibatch(doc_stream, size):
        docs, y = [], []
        try:
            for _ in range(size):
                doc, label = next(doc_stream)
                docs.append(doc)
                y.append(label)
        except StopIteration:
            return None, None
        return docs, y

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    import pickle
    bar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    vect = HashingVectorizer(decode_error='ignore', n_features= 2 ** 21, preprocessor=None, tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', max_iter=1, random_state=1)
    doc_stream = stream_docs(path)

    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        bar.update()

    X_test, y_test = get_minibatch(doc_stream, 5000)
    X_test = vect.transform(X_test)
    print('Accuracy:%.3f '% clf.score(X_test, y_test))

    # pickle.dump(stop, open(r'E:\PythonWB\web\data\stopwords.pkl', 'wb'))
    # pickle.dump(clf, open(r'E:\PythonWB\web\data\classifier.pkl', 'wb'))


if __name__ == "__main__":
    path = 'data/imdb.csv'
    """
    # process_data()
    df = pd.read_csv(path)
    print(df.loc[5, 'review'][-50:])
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values
    # optimize(X_train, y_train)
    # best_model(X_train, y_train, X_test, y_test)
    """
    test_waicun_learning(path)