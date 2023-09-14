# not all imports are used here, we should clean them up before delivering
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
    # Load saved dataet into df
    df = pd.DataFrame(load_csv(path), columns=["Label", "Sentence"])

    # split the dataset into training 85% and testing 15%
    X = df["Sentence"]
    y = df["Label"]

    # the random state is that the split is done in the same way every time the code is being run
    # so that we don t have to make separate files for the data splits
    return train_test_split(X, y, test_size=0.15, random_state=42)


def train_and_evaluate(path):
    # Get ready data from path of saved file
    X_train, X_test, y_train, y_test = process(path)

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
    # Paths of the stored data
    data_path = "./data/dialog_acts.csv"
    data_path_dedup = "./data/dialog_acts_dedup.csv"

    # Run code for both datasets
    print("complete dataset:")
    train_and_evaluate(data_path)

    print("deleted duplicates dataset:")
    train_and_evaluate(data_path_dedup)
