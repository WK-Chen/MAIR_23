import os
import sys
from matplotlib import pyplot as plt
import re

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
sys.path.append(os.getcwd())
from utils.utils import *


def train_and_evaluate(path):
    def baseline_classifier2(it):
        if it.find("thank") != -1:
            return "thankyou"
        elif it.find("bye") != -1:
            return "bye"
        elif it.find("about") != -1:
            return "reqalts"
        elif (it.find("phone") != -1 or it.find("post") != -1 or it.find("adress") != -1 or it.find("area") != -1 or it.find("located") != -1 or it.find("range") != -1 or it.find("type") != -1) or (it.find("what") != -1 or it.find("can") != -1) or it.find("how") != -1:
            return "request"  
        elif it.find("yes") != -1 or it.find("yeah") != -1 or it.find("okay") != -1:
            return "affirm"
        elif bool(re.search(r'\bno\b|^no\b|\bno$', it)):
            return "negate"
        elif it.find("unintelligible") != -1 or it.find("unintelligible") != -1 or it.find("cough") != -1 or it.find("sil") != -1 or it.find("noise") != -1 or it.find("inaudible") != -1:
            return "null"
        elif it.find("does it") != -1 or it.find("is it") != -1 or it.find("is that") != -1 or it.find("is there") != -1:
            return "confirm"
        elif it.find("repeat") != -1:
            return "repeat"
        elif it.find("start") != -1 or it.find("reset") != -1:
            return "restart"
        elif it.find("dont") != -1 or it.find("wrong") != -1 or it.find("not") != -1 or it.find("else") != -1:
            return "deny"
        elif it.find("hi") != -1 or it.find("hello") != -1 or it.find("halo") != -1:
            return "hello"
        else:
            return "inform"

    # Get ready data from path of saved file
    X_train, X_test, y_train, y_test = split_dataset_pd(path)

    # Test any baseline classifier on the test data
    correct_predictions = 0
    total_predictions = len(y_test)

    for sentence, true_label in zip(X_test, y_test):
        # the next 2 lines are for the 2nd classifier
        predicted_label = baseline_classifier2(sentence)
        if predicted_label == true_label:
            correct_predictions += 1

    print(f"Prediction accuracy of model: {correct_predictions / total_predictions}")
    
    y_predicted = X_train.apply(lambda x: baseline_classifier2(x))
    
    # Print the confusion matrix
    ul = y_train.unique()
    confusion = confusion_matrix(y_train, y_predicted,labels=ul)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=ul,
    )
    disp.plot()
    if path == data_path:
        plt.figure(1)
        plt.title("Figure 1: Confusion Matrix of the dataset with duplicates")
    elif path == data_path_dedup:
        plt.figure(2)
        plt.title("Figure 2: Confusion Matrix of the dataset without duplicates")
    

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Paths of the stored data
    data_path = os.path.join(root_path, "data/dialog_acts.csv")
    data_path_dedup = os.path.join(root_path, "data/dialog_acts_dedup.csv")

    # Run code for both datasets
    print("complete dataset:")
    train_and_evaluate(data_path)

    print("deleted duplicates dataset:")
    train_and_evaluate(data_path_dedup)

    plt.show()
