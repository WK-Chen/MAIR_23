import os
import sys
sys.path.append(os.getcwd())
from utils.utils import *


def train_and_evaluate(path):
    def baseline_classifier2(it):
        if it.find("thank") != -1:
            return "thankyou"
        elif it.find("food") != -1:
            return "inform"
        elif it.find("else") != -1:
            return "reqalts"
        elif it.find("about") != -1:
            return "reqalts"
        elif it.find("what") != -1:
            return "request"
        elif it.find("phone") != -1:
            return "request"
        elif it.find("adress") != -1:
            return "request"
        elif it.find("post") != -1:
            return "request"
        elif it.find("type") != -1:
            return "request"
        elif it.find("kind") != -1:
            return "request"
        elif it.find("price") != -1:
            return "request"
        elif it.find("where") != -1:
            return "request"
        elif it.find("area") != -1:
            return "request"
        elif it.find("unintelligible") != -1:
            return "null"
        elif it.find("cough") != -1:
            return "null"
        elif it.find("sil") != -1:
            return "null"
        elif it.find("noise") != -1:
            return "null"
        elif it.find("inaudible") != -1:
            return "null"
        elif it.find("yes") != -1:
            return "affirm"
        else:
            return "inform"

    # Get ready data from path of saved file
    X_train, X_test, y_train, y_test = split_dataset_pd(path)

    # Test any baseline classifier on the test data
    correct_predictions = 0
    total_predictions = len(y_test)

    y_predicted = X_train.apply(lambda x: baseline_classifier2(x))

    for sentence, true_label in zip(X_test, y_test):
        # the next 2 lines are for the 2nd classifier
        predicted_label = baseline_classifier2(sentence)
        if predicted_label == true_label:
            correct_predictions += 1

    print(f"Prediction accuracy of model: {correct_predictions / total_predictions}")

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


    # print(y_train.value_counts(), y_predicted.value_counts())
    #
    # ul = y_train.unique()
    #
    # confusion = confusion_matrix(y_train, y_predicted,labels=ul)
    #
    #
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=confusion,
    #     display_labels=ul,
    # )
    # disp.plot()
    # plt.show()
