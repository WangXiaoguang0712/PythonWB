# _*_ coding:utf-8 _*_
import jieba
import numpy as np
import chardet
__author__ = 'T'

class NGram():
    def __init__(self, n = 2):
        if n > 3:
            raise 'the figure for n-gram is tremendous'
            return
        else:
            self.n = n
        self.core = {'unigram':{}, 'bigram':{}, 'trigram':{}}

    def segment(self, line):
        line = line.strip().rstrip('。').replace('。', 'EOSBOS').decode('utf-8-sig')
        paragraph = 'BOS' + line + 'EOS'
        print(paragraph)
        # print(paragraph.decode('utf-8'))
        jieba.suggest_freq('BOS', True)
        jieba.suggest_freq('EOS', True)
        return ','.join(jieba.cut(paragraph))

    def save_core(self, sent):
        print(sent)
        if self.n >= 1:
            udict = self.core['unigram']
            for x in sent.split(','):
                if x not in udict:
                    udict[x] = 1
                else:
                    udict[x] += 1
            self.core['unigram']

        if self.n >= 2:
            bdict = self.core['bigram']
            l_word = sent.split(',')
            for i in range(len(l_word) - 1):
                # 判断当前词是否在字典中
                if l_word[i] not in bdict:
                    bdict[l_word[i]] = {}
                # 判断下一个词是否在子字典中
                if l_word[i + 1] not in bdict[l_word[i]]:
                    bdict[l_word[i]][l_word[i + 1] ] = 1
                else:
                    bdict[l_word[i]][l_word[i + 1] ] += 1

        if self.n == 3:
            tdict = self.core['trigram']
            l_word = sent.split(',')
            for i in range(len(l_word) - 1):
                if l_word[i] not in bdict:
                    bdict[l_word[i]] = {}
                if l_word[i + 1] not in bdict[l_word[i]]:
                    bdict[l_word[i]][l_word[i + 1]] = {}
                if l_word[i + 2] not in bdict[l_word[i]][l_word[i + 1]]:
                    bdict[l_word[i]][l_word[i + 1]][l_word[i + 2]] = 1
                else:
                    bdict[l_word[i]][l_word[i + 1]][l_word[i + 2]] += 1

    def learn(self, corpus):
        with open(corpus,'r') as f:
            for line in f.readlines():
                self.save_core(self.segment(line))
        print(self.core['bigram'])
        print(self.core['unigram'])

    def good_turing(self, tag):
        word_1, word_all = 0, 0
        if tag == 'unigram':
            word_all = sum([x[1] for x in self.core['unigram'].items()])
            word_1 = sum([ 1 if x[1] == 1 else 0 for x in self.core['unigram'].items()])
            res = word_1 * 1. / word_all
        elif tag == 'bigram':
            for x in self.core['bigram'].items():
                for y in  x[1].items():
                    word_all += 1
                    if y[1] == 1:
                        word_1 += 1
            res = word_1 * 1. / word_all
            print(word_1)
        return res

    def calc_probability(self, sents):
        sents = self.segment(sents).split(',')
        p = 1.0
        for i in range(1,len(sents)):
            bi, uni = 0, 0
            try:
                bi = self.core['bigram'][sents[i - 1]][sents[i]]
            except:
                pass
            try:
                uni = self.core['unigram'][sents[i]]
            except:
                pass
            print(sents[i],bi,uni)
            p *= float(bi + 1) / float(uni + 1)
        return p


if __name__ == '__main__':

    corpus = 'data/ngram-corpus.txt'
    sent = '他是研究生物的'
    gm = NGram()
    gm.learn(corpus)
    print gm.calc_probability(sent)