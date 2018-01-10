import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
__author__ = 'T'


class Language_Detector():
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.victorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)

    def _remove_noise(self, doc):
        noise_pat = re.compile('|'.join(['http:\S+','\@\w+','\#\w+']))
        clean_text = re.sub(noise_pat, '', doc)
        return  clean_text

    def features(self, X):
        return self.victorizer.transform(X)

    def fit(self, X, y):
        self.victorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


if __name__ == '__main__':
    with open('data/data.csv') as f:
        dataset = [(line.strip()[:-3],line.strip()[-2:])  for line in f.readlines()]
    X, y = zip(*dataset)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    ld = Language_Detector()
    ld.fit(x_train, y_train)
    print(ld.score(x_test, y_test))
    print(ld.predict('the'))

