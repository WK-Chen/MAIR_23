# not all imports are used here, we should clean them up before delivering
from matplotlib import pyplot as plt
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


def baseline_classifier2(it):
    if it.find("food"):
        return "inform"
    return "T"


# Define paths
data_path = "./data/dialog_acts.csv"
data_path_dedup = "./data/dialog_acts_dedup.csv"

X_train, X_test, y_train, y_test = split_dataset_pd(data_path)

# Test any baseline classifier on the test data
correct_predictions = 0
total_predictions = len(y_test)

y_predicted = X_train.apply(lambda x: baseline_classifier2(x))

for sentence, true_label in zip(X_test, y_test):
    # the next 2 lines are for the 2nd classifier
    predicted_label = baseline_classifier2(sentence)
    if predicted_label == true_label:
        correct_predictions += 1

print(f"Accuracy on test data: {correct_predictions / total_predictions}")


print(y_train.value_counts(), y_predicted.value_counts())


confusion = confusion_matrix(y_train, y_predicted)


disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=set(pd.concat([y_train.drop_duplicates(), pd.Series(["T"])])),
)
disp.plot()
plt.show()
