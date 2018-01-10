# _*_ coding:utf-8 _*_
import jieba
import jieba.analyse

# 停用词
def f4():
    content = '如果放到post中将出错。'
    # jieba.analyse.set_stop_words("data/stopwords.txt")
    tags = jieba.analyse.extract_tags(content, topK=20)
    #jieba.del_word('出错')
    print(",".join(tags))
    print(" ".join(jieba.cut(content)))

# 返回词语在原文的起止位置
def f5():
    result = jieba.tokenize(u'永和服装饰品有限公司')
    for tk in result:
        print("word %s\t\t start:%s\t\t end:%s" %(tk[0], tk[1], tk[2]))

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# 分句
def segmenter():
    text = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
    sents = sent_tokenize(text)
    for s in sents:
        print s
        words = nltk.word_tokenize(s)
        print words
        print nltk.pos_tag(words)

# 通过名字预测性别
from nltk.corpus import names
from nltk import classify
def gender_features(word):
    return {'last_letter': word[-1]}

def predict_main():
    name = [(n,'male') for n in names.words('male.txt')]+[(n,'female') for n in names.words('female.txt')]
    # print len(name)
    features = [ (gender_features(n),g) for n,g in name]
    # print features[:100]
    classifier = nltk.NaiveBayesClassifier.train(features[:6000])
    print(classifier.classify(gender_features('Frank')))
    print(classify.accuracy(classifier,features[6000:]))


# 情感分析
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def word_feats(words):
    return dict([(word,True) for word in words])


def feeling_main():
    # 数据准备
    positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
    negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
    neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']
    # 特征提取
    positive_features = [(word_feats(w),'pos') for w in positive_vocab]
    negative_features = [(word_feats(w),'neg') for w in negative_vocab]
    neutral_features = [(word_feats(w),'neu') for w in neutral_vocab]
    #print positive_feature
    train_set = negative_features + positive_features + neutral_features
    # 训练
    classifier = NaiveBayesClassifier.train(train_set)
    # 测试
    neg = 0
    pos = 0
    sentence = "bad movie, I do not hate it"
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        classResult = classifier.classify(word_feats(word))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
    print('Positive: ' + str(float(pos) / len(words)))
    print('Negative: ' + str(float(neg) / len(words)))

if __name__ == "__main__":
    feeling_main()