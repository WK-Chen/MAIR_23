import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import *


def process(path):
    data = load_csv(path)

    # here the label column is for the type of dialog act and the sentence column for the dialog acts
    df = pd.DataFrame(data, columns=['Label', 'Sentence'])

    # split the dataset into training 85% and testing 15%
    X = df['Sentence'].astype(str)
    y = df['Label']
    # print(y) - this was just to check that the labels are split correctly

    X_train, X_test = X[:int(len(X)*0.85)], X[int(len(X)*0.85):]
    y_train, y_test = y[:int(len(X)*0.85)], y[int(len(X)*0.85):]

    return X_train, X_test, y_train, y_test


def train(classifier, X_train, y_train):
    """KNN has the best accuracy - 0.97, but all that I tested were good >0.94 but I m not sure ab the other metrics,
    it might do overfitting you can test with any classifier from scikit learn by importing the model above and then
    using it here directly, but I don t think we should focus on that
    I think we should include all the relevant metrics first so we can have some conclusions for the models"""

    X_train = vectorizer.fit_transform(X_train)
    classifier.fit(X_train, y_train)

    return classifier


def evaluate(classifier, X_test, y_test):
    X_test = vectorizer.transform(X_test)
    y_predicted = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(f'Accuracy on test data: {accuracy}')


if __name__ == '__main__':
    # bag of words
    vectorizer = CountVectorizer()

    # classifier = MultinomialNB()
    # classifier =   SVC(kernel="linear", C=0.025)
    classifier = KNeighborsClassifier(3)

    X_train, X_test, y_train, y_test = process("../data/dialog_acts.csv")
    classifier = train(classifier, X_train, y_train)
    evaluate(classifier, X_test, y_test)
