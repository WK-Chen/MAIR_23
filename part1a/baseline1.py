import os
import sys
from utils.utils import *


def train_and_evaluate(path):
    # Get ready data from path of saved file
    X_train, X_test, y_train, y_test = split_dataset_pd(path)

    # Determine the majority class label
    majority_class = y_train.value_counts().idxmax()

    # Function that always predicts the majority class label
    def baseline_classifier1(input_text):
        return majority_class

    # Test any baseline classifier on the test data
    correct_predictions = 0
    total_predictions = len(y_test)

    for sentence, true_label in zip(X_test, y_test):
        if true_label == baseline_classifier1(sentence):
            correct_predictions += 1

    print(f"Prediction accuracy of model: {correct_predictions / total_predictions}")


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Paths of the stored data
    data_path = os.path.join(root_path, "data/dialog_acts.csv")
    data_path_dedup = os.path.join(root_path, "data/dialog_acts_dedup.csv")

    # Run code for both datasets
    print("complete dataset:")
    train_and_evaluate(data_path)

    print("deleted duplicates dataset:")
    train_and_evaluate(data_path_dedup)
