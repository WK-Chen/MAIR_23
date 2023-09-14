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

# open and read file
with open('dialog_acts.csv', 'r') as file:
    lines = file.readlines()

# this is to separate the label and sentence correctly, had some problems on some lines without this step
data = []
for line in lines:
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        label, sentence = parts
        data.append((label.lower(), sentence))

# here the label column is for the type of dialog act and the sentence column for the dialog acts
df = pd.DataFrame(data, columns=['Label', 'Sentence'])

# lowered the sentences but this did not make any difference in the prediction till now
df['Sentence'] = df['Sentence'].str.lower()

# lower labels
df['Label'] = df['Label'].str.lower()

# split the dataset into training 85% and testing 15% 
X = df['Sentence'].astype(str)
y = df['Label']

#print(y) - this was just to check that the labels are split correctly
# the random state is that the split is done in the same way every time the code is being run
# so that we don t have to make separate files for the data splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Determine the majority class label
majority_class = y.value_counts().idxmax()

# Function that always predicts the majority class label
def baseline_classifier1(input_text):
    return majority_class

# Test any baseline classifier on the test data
correct_predictions = 0
total_predictions = len(y_test)

for sentence, true_label in zip(X_test, y_test):
 if true_label == 'inform':  
        correct_predictions += 1

print(f'Accuracy on test data: {correct_predictions / total_predictions}')
