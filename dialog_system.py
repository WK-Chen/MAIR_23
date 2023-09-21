from classifier1 import *

data_path = "./data/dialog_acts.csv"
data_path_dedup = "./data/dialog_acts_dedup.csv"

vectorizer = CountVectorizer()
classifier = KNeighborsClassifier(3)
X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
classifier = train(vectorizer, classifier, X_train, y_train)

def prediction(utterance: str):
    return predict(utterance, classifier, vectorizer)


def state_transition(dialog_state, user_utterance):
    