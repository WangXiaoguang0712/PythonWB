# _*_ coding:utf-8 _*_
__author__ = 'T'

import sys
sys.path.append('..')
from flask import Flask
from flask import render_template
from flask import request
import pickle
from web.vectorizer import vect
from web.dbhelper import DBHelper
import numpy as np

app = Flask(__name__)
clf = pickle.load(open(r'E:\PythonWB\web\data\classifier.pkl', 'rb'))
db_path = 'data/data.db'

@app.route('/')
def index():
    return render_template('first_app.html')


@app.route('/appraisal', methods=['POST'])
def appraisal():
    form = request.form
    _res, _prob = classify(form['review'])
    print(_res, _prob)
    return render_template('appraisal.html', review=form['review'], res=(_res, format(_prob*100,'0.2f')))

@app.route('/thanks', methods=['POST'])
def thanks():
    feedback = request.form['feedback']
    review = request.form['review']
    predict = request.form['predict']
    inv_label = {'负面的':0, '正面的':1}
    y = inv_label[predict]
    if feedback == '不正确':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db_path, review, y)
    return render_template('thanks.html')

def classify(text):
    labels = {0: '负面的', 1: '正面的'}
    # example = ['I dislike this movie']
    X = vect.transform([text])
    # print('prediction: %s \nprobobality: %.2f%%' % (labels[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))
    return labels[clf.predict(X)[0]], np.max(clf.predict_proba(X))

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    db = DBHelper(db_path=path)
    db.exec('insert into review_db values (?,?,datetime("now"))', (document, y))

if __name__ == "__main__":
    app.run(debug=True)