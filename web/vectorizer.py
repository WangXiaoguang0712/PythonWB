# _*_ coding:utf-8 _*_
__author__ = 'T'

import re
import os
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

curr_dir = os.path.dirname(__file__)
stop = pickle.load(open(r'E:\PythonWB\web\data\stopwords.pkl', 'rb'))

def tokennizer(str):
    text = re.sub('<[^>]*>', '', str)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
    tokenized = [ w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ingore', n_features= 2**21, preprocessor=None, tokenizer=tokennizer)

if __name__ == "__main__":
    pass