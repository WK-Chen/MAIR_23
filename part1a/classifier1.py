import os
import sys
sys.path.append(os.getcwd())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


from utils.utils import *

def train(vectorizer, classifier, X_train, y_train):
    """KNN has the best accuracy - 0.97, but all that I tested were good >0.94 but I m not sure ab the other metrics,
    it might do overfitting you can test with any classifier from scikit learn by importing the model above and then
    using it here directly, but I don t think we should focus on that
    I think we should include all the relevant metrics first so we can have some conclusions for the models
    """

    X_train = vectorizer.fit_transform(X_train)
    classifier.fit(X_train, y_train)
    return classifier


def evaluate(vectorizer, classifier, X_test, y_test, classes,path):
    X_test = vectorizer.transform(X_test)
    y_predicted = classifier.predict(X_test)

    # Print evaluation metrics
    print(f"Accuracy on test data: {accuracy_score(y_test, y_predicted)}")
    print(f"Average precision score: {precision_score(y_test, y_predicted, average='macro', zero_division=1.0)}")
    print(f"Average recall score: {recall_score(y_test, y_predicted, average='macro', zero_division=1.0)}")
    print(f"Average F1 score score: {f1_score(y_test, y_predicted, average='macro', zero_division=1.0)}")
    
    # Print Condusion Matrix
    confusion = confusion_matrix(y_test, y_predicted, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classes)
    disp.plot()
    # plt.show(block=False)
    if path == data_path:
        plt.figure(1)
        plt.title("Figure 1: Confusion Matrix of the dataset with duplicates")
    elif path == data_path_dedup:
        plt.figure(2)
        plt.title("Figure 2: Confusion Matrix of the dataset without duplicates")

def interaction(vectorizer, classifier):
    # Start diaglogue
    while True:
        user_input = input("User: ")
        if user_input == "":
            break
        print(predict(user_input, classifier, vectorizer))

def predict(utterance : str, classifier, vectorizer):
    return classifier.predict(vectorizer.transform([utterance]))[0]


if __name__ == "__main__":
    # Get the dataset
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_path, sys.argv[1])
    data_path_dedup = os.path.join(root_path, "data/dialog_acts_dedup.csv")

    # Make it into train-test sets
    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)

    # Create and train classifier
    vectorizer = CountVectorizer()
    classifier = KNeighborsClassifier(3)

    classifier = train(vectorizer, classifier, X_train, y_train)

    # # Evaluate performace
    # evaluate(vectorizer, classifier, X_test, y_test, classifier.classes_)

    # Run code for both datasets
    print("complete dataset:")
    # Make it into train-test sets
    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
    classifier = train(vectorizer, classifier, X_train, y_train)
    # Evaluate performace
    evaluate(vectorizer, classifier, X_test, y_test, classifier.classes_, data_path)

    print("deleted duplicates dataset:")
    X_train_1, X_test_1, y_train_1, y_test_1 = split_dataset_pd(data_path_dedup)
    classifier = train(vectorizer, classifier, X_train_1, y_train_1)
    # Evaluate performace
    evaluate(vectorizer, classifier, X_test_1, y_test_1, classifier.classes_, data_path_dedup)
    plt.show()

    # # Predict with user input
    # interaction(vectorizer, classifier)
