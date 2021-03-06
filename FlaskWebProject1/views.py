"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,request
from FlaskWebProject1 import app
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def machine_model(train_data):
    dataset = pd.read_csv(train_data)
    dataset .columns 
    dataset.shape  

    dataset.drop_duplicates(inplace = True)
    dataset.shape  

    dataset['text']=dataset['text'].map(lambda text: text[1:])
    dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
    ps = PorterStemmer()
    corpus=dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))

    cv = CountVectorizer()
    X = cv.fit_transform(corpus.values).toarray()
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    classifier.fit(X_train , y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    total=cm[0][0]+cm[0][0]
    wrong=cm[0][1]+cm[1][0]
    ac_score=accuracy_score(y_test, y_pred,normalize=False) 
    number=len(y_test)
    return (((total-wrong)/total)*100),ac_score,number

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact', methods = ['POST'])  
def contact():  
    if request.method == 'POST':  
        global f
        f = request.files['file']  
        f.save(f.filename)  
        training_data=f.filename
        result,ac,number=machine_model(training_data)
        return render_template('contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.', name = f.filename,results=result,accuracy=ac,numbers=number)
