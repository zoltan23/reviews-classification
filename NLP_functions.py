import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

def cleanData(dataStr):
    dataStr = pd.Series(dataStr)
    corpus = []
    for i in range(0, len(dataStr.index)):
        review = re.sub('[^a-zA-Z]', ' ', dataStr[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

from sklearn.neighbors import KNeighborsClassifier
def createKNN(X_train, y_train, n_neighbors, metric, p):
    classifier = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric, p = p)
    classifier.fit(X_train, y_train)
    return classifier

from sklearn.linear_model import LogisticRegression
def createLogistic(X_train, y_train, random_state):
    classifier = LogisticRegression(random_state = random_state)
    classifier.fit(X_train, y_train)
    return classifier

from sklearn.naive_bayes import GaussianNB
def createNaiveBayes(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

from sklearn.ensemble import RandomForestClassifier
def createRandomForest(X_train, y_train, n_estimators):
    classifier = RandomForestClassifier(n_estimators = n_estimators)
    classifier.fit(X_train, y_train)
    return classifier

from sklearn.svm import SVC
def createSVM(X_train, y_train, kernel, random_state):
    classifier = SVC(kernel = kernel, random_state = random_state)
    classifier.fit(X_train, y_train)
    return classifier

def createPickleModel(classifier, model_name):
    pickle.dump(classifier, open(model_name + '.pkl', 'wb'))
    model = pickle.load(open(model_name + '.pkl', 'rb'))
    return model