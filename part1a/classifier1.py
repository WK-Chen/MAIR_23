import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from utils import *


# def process(path):
#     # Load saved dataet into df
#     df = pd.DataFrame(load_csv(path), columns=["Label", "Sentence"])
#
#     # split the dataset into training 85% and testing 15%
#     X = df["Sentence"]
#     y = df["Label"]
#
#     # the random state is that the split is done in the same way every time the code is being run
#     # so that we don t have to make separate files for the data splits
#     return train_test_split(X, y, test_size=0.15, random_state=42)


def train(vectorizer, classifier, X_train, y_train):
    """KNN has the best accuracy - 0.97, but all that I tested were good >0.94 but I m not sure ab the other metrics,
    it might do overfitting you can test with any classifier from scikit learn by importing the model above and then
    using it here directly, but I don t think we should focus on that
    I think we should include all the relevant metrics first so we can have some conclusions for the models
    """

    X_train = vectorizer.fit_transform(X_train)
    classifier.fit(X_train, y_train)

    return classifier


def evaluate(vectorizer, classifier, X_test, y_test, classes):
    X_test = vectorizer.transform(X_test)
    y_predicted = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy on test data: {accuracy}")

    confusion = confusion_matrix(y_test, y_predicted, labels=classes)
    print(confusion)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classes)
    disp.plot()
    plt.show()

def interaction(vectorizer, classifier):
    # Start diaglogue
    while True:
        user_input = input("User: ")
        if user_input == "":
            break
        print(predict(user_input))

def predict(utterance : str, classifier, vectorizer):
    return classifier.predict(vectorizer.transform([utterance]))[0]


if __name__ == "__main__":
    # Paths of the stored data
    data_path = "../data/dialog_acts.csv"
    data_path_dedup = "../data/dialog_acts_dedup.csv"

    # bag of words
    vectorizer = CountVectorizer()

    # classifier = MultinomialNB()
    # classifier =   SVC(kernel="linear", C=0.025)
    classifier = KNeighborsClassifier(3)

    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)

    classifier = train(vectorizer, classifier, X_train, y_train)
    evaluate(vectorizer, classifier, X_test, y_test, classifier.classes_)

    # predict with human input
    interaction(vectorizer, classifier)