# _*_ coding:utf-8 _*_
__author__ = 'T'

from flask import Flask
from flask import render_template
from flask import request
import pickle
from web.vectorizer import vect
from web.dbhelper import DBHelper
import numpy as np

app = Flask(__name__)
clf = pickle.load(open(r'E:\PythonWB\web\data\classifier.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('first_app.html')


@app.route('/appraisal', methods=['POST'])
def appraisal():
    form = request.form
    print(form)
    _review = form['review']

    return render_template('appraisal.html', review=_review)


def classify():
    labels = {0: 'negative', 1: 'positinve'}
    example = ['I dislike this movie']
    X = vect.transform(example)
    # print('prediction: %s \nprobobality: %.2f%%' % (labels[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
    return clf.predict(X)[0], np.max(clf.predict_proba(X))

def train(document, y):
    X = vect.transform(document)
    clf.partial_fit(X, y)

def sqlite_entry(path, document, y):
    db = DBHelper(db_path=path)
    db.exec('insert into review_db values (?,?,datetime("now"))', (document, y))

if __name__ == "__main__":
    app.run(debug=True)