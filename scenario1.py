import os
import sys
sys.path.append(os.getcwd())
import time
from transitions import Machine

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import csv
import pandas as pd
import Levenshtein

# Open the CSV file and return its content as an array
def load_csv(path: str):
    data = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader]
    return data


# Write the data form an array to a CSV file
def save_csv(path: str, data):
    with open(path, "w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)
    print(f"Conversion complete. Data saved in '{path}'.")

def split_dataset_pd(path):
    # Load saved dataet into df
    df = pd.DataFrame(load_csv(path), columns=["Label", "Sentence"])

    # split the dataset into training 85% and testing 15%
    X = df["Sentence"]
    y = df["Label"]

    length = len(X)
    # separate using percentage 0.85/0.15
    return X[:int(length*0.85)], X[int(length*0.85):], y[:int(length*0.85)], y[int(length*0.85):]

def find_nearest_keyword(user_utterance, keyword_list):
    # Initialize variables to store the closest keyword and its Levenshtein distance
    closest_keyword = None
    min_distance = float('inf')  # Initialize to positive infinity

    # Split the user's input sentence into individual words
    user_words = user_utterance.split()

    # Iterate through each word in the user's input
    for user_word in user_words:
        # Iterate through the keyword list for this category
        for keyword in keyword_list:
            # Calculate the Levenshtein distance between the user's word and the keyword
            distance = Levenshtein.distance(user_word, keyword.lower())

            # Check if the current distance is smaller than the minimum distance found so far
            if distance < min_distance:
                min_distance = distance
                closest_keyword = keyword

    # Return the closest keyword and its Levenshtein distance
    return closest_keyword, min_distance

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
    _X_test = X_test
    X_test = vectorizer.transform(X_test)
    y_predicted = classifier.predict(X_test)

    # for xt, yt, yp in zip(_X_test, y_test, y_predicted):
    #     if yt != yp:
    #         print(f"{xt} | {yt} | {yp}")

    # Print evaluation metrics
    print(f"Accuracy on test data: {accuracy_score(y_test, y_predicted)}")
    print(f"Average precision score: {precision_score(y_test, y_predicted, average='macro', zero_division=1.0)}")
    print(f"Average recall score: {recall_score(y_test, y_predicted, average='macro', zero_division=1.0)}")
    print(f"Average F1 score score: {f1_score(y_test, y_predicted, average='macro', zero_division=1.0)}")

    # Print Condusion Matrix
    confusion = confusion_matrix(y_test.tolist(), y_predicted, labels=classes)
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

# Keywords for prefer extraction
food_type = ["british", "modern european", "italian", "romanian", "chinese", "seafood", "steakhouse", "asian oriental",
             "french", "chinese", "portuguese", "indian", "spanish", "japanese", "european", "vietnamese", "korean",
             "thai", "moroccan", "swiss", "fusion", "gastropub", "tuscan", "international", "traditional",
             "mediterranean", "polynesian", "african", "turkish", "north american", "australasian", "persian",
             "jamaican", "lebanese", 'cuban', "catalan"]
area = ["east", "south", "centre", "north", "west"]
price_range = ["cheap", "moderate", "expensive"]
additional = ["touristic", "assigned", "children", "romantic", "long", "short", "good", "bad", "busy", "leisure"]

def prediction(classifier, vectorizer, utterance: str):
    res = predict(utterance, classifier, vectorizer)
    return res

def search_restaurants(data, filter):
    # lookup function to search restaurants from database
    res = data[
        (data['pricerange'] == filter['price_range']) &
        (data['area'] == filter['area']) &
        (data['food'] == filter['food_type'])
        ]
    return res

def filter_restaurants(data, filter):
    if filter['addition'][-1] == "touristic":
        res = data[
            (data['food_quality'] == "good") &
            (data['pricerange'] == "cheap") &
            (data['food'] != "romanian")
            ]
    elif filter['addition'][-1] == "assigned_seats":
        res = data[(data['crowdedness'] == "busy")]
    elif filter['addition'][-1] == "children":
        res = data[(data['length_of_stay'] != "long")]
    elif filter['addition'][-1] == "romantic":
        res = data[
            (data['crowdedness'] != "busy") &
            (data['length_of_stay'] == "long")
            ]
    elif filter['addition'][-1] in ["good", "bad"]:
        res = data[(data['food_quality'] == filter['addition'][-1])]
    elif filter['addition'][-1] in ["busy", "leisure"]:
        res = data[(data['crowdedness'] == filter['addition'][-1])]
    elif filter['addition'][-1] in ["short", "long"]:
        res = data[(data['length_of_stay'] == filter['addition'][-1])]
    else:
        return data
    return res
class Dialog:
    def __init__(self, classifier, vectorizer, restaurants):
        # Define the states
        states = ['start', 'welcome', 'ask_delay', 'ask_caps',
                  'ask_area', 'area_confirm', 'ask_price', 'price_confirm', 'ask_food_type', 'food_confirm',
                  'check_recommend',  'ask_start_over',
                  'ask_addition', 'addition_confirm', 'filter_recommend', 'ask_reset', 'give_recommend',
                  'detail', 'end']

        # Define the transitions
        transitions = [
            {'trigger': 'forward', 'source': 'start', 'dest': 'welcome'},
            {'trigger': 'forward', 'source': 'welcome', 'dest': 'ask_delay'},

            {'trigger': 'forward', 'source': 'ask_delay', 'dest': 'ask_caps'},
            {'trigger': 'again', 'source': 'ask_delay', 'dest': '='},

            {'trigger': 'forward', 'source': 'ask_caps', 'dest': 'ask_area'},
            {'trigger': 'again', 'source': 'ask_caps', 'dest': '='},

            {'trigger': 'forward', 'source': 'ask_area', 'dest': 'ask_price'},
            {'trigger': 'confirm', 'source': 'ask_area', 'dest': 'area_confirm'},
            {'trigger': 'again', 'source': 'ask_area', 'dest': '='},

            {'trigger': 'forward', 'source': 'area_confirm', 'dest': 'ask_price'},
            {'trigger': 'backward', 'source': 'area_confirm', 'dest': 'ask_area'},

            {'trigger': 'forward', 'source': 'ask_price', 'dest': 'ask_food_type'},
            {'trigger': 'confirm', 'source': 'ask_price', 'dest': 'price_confirm'},
            {'trigger': 'again', 'source': 'ask_price', 'dest': '='},

            {'trigger': 'forward', 'source': 'price_confirm', 'dest': 'ask_food_type'},
            {'trigger': 'backward', 'source': 'price_confirm', 'dest': 'ask_price'},

            {'trigger': 'forward', 'source': 'ask_food_type', 'dest': 'check_recommend'},
            {'trigger': 'confirm', 'source': 'ask_food_type', 'dest': 'food_confirm'},
            {'trigger': 'again', 'source': 'ask_price', 'dest': '='},

            {'trigger': 'forward', 'source': 'food_confirm', 'dest': 'check_recommend'},
            {'trigger': 'backward', 'source': 'food_confirm', 'dest': 'ask_food_type'},

            # new here
            {'trigger': 'success', 'source': 'check_recommend', 'dest': 'ask_addition', 'before': 'set_delay'},
            {'trigger': 'failure', 'source': 'check_recommend', 'dest': 'ask_start_over', 'before': 'set_delay'},

            {'trigger': 'forward', 'source': 'ask_start_over', 'dest': 'end'},
            {'trigger': 'again', 'source': 'ask_start_over', 'dest': 'welcome'},

            {'trigger': 'forward', 'source': 'ask_addition', 'dest': 'filter_recommend'},
            {'trigger': 'skip', 'source': 'ask_addition', 'dest': 'give_recommend'},
            {'trigger': 'confirm', 'source': 'ask_addition', 'dest': 'addition_confirm'},
            {'trigger': 'again', 'source': 'ask_addition', 'dest': '='},

            {'trigger': 'forward', 'source': 'addition_confirm', 'dest': 'filter_recommend'},
            {'trigger': 'backward', 'source': 'addition_confirm', 'dest': 'ask_addition'},

            # {'trigger': 'success', 'source': 'filter_recommend', 'dest': 'give_recommend', 'before': 'set_delay'},
            {'trigger': 'success', 'source': 'filter_recommend', 'dest': 'ask_addition', 'before': 'set_delay'},
            {'trigger': 'failure', 'source': 'filter_recommend', 'dest': 'ask_reset', 'before': 'set_delay'},

            {'trigger': 'success', 'source': 'ask_reset', 'dest': 'ask_addition'},
            {'trigger': 'failure', 'source': 'ask_reset', 'dest': 'ask_start_over'},

            {'trigger': 'success', 'source': 'give_recommend', 'dest': 'detail', 'before': 'set_delay'},
            {'trigger': 'failure', 'source': 'give_recommend', 'dest': 'check_recommend'},

            {'trigger': 'forward', 'source': 'detail', 'dest': 'end'},
        ]

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, transitions=transitions, initial='start')

        # Define the classifier
        self.classifer = classifier
        self.vectorizer = vectorizer

        # Initialize current status and confirm text
        self.status = "forward"
        self.confirm_text = ""

        # Load data, Initialize filter, first search, recommend list and filter list
        self.restaurants = restaurants
        self.filter = {"area": "", "price_range": "", "food_type": "", "addition": []}
        self.first_time = True
        self.recommend_list = pd.DataFrame()
        self.filter_list = pd.DataFrame()

        # Initialize switches
        self.delay = False
        self.caps = False

    def get_user_input(self, field=None):
        message = input("User: ").lower()
        act = prediction(self.classifer, self.vectorizer, message)
        # print(act)
        if act == "null" and field:
            closest_key, min_distance = find_nearest_keyword(message, field)
            act = prediction(self.classifer, self.vectorizer, closest_key)
        return message, act

    def set_delay(self):
        if self.delay:
            print("Now searching ...")
            time.sleep(3)
    def capitalize(self, input):
        return input.upper() if self.caps else input


    def print_restaurant(self, data, detail=False):
        info = f"I recommend {data[0]}, it is an {data[1]} {data[3]} restaurant in the {data[2]} of town."
        addition = ""
        for feature in self.filter['addition']:
            if feature in ["good", "bad"]:
                addition = f"It is a restaurant in {feature} food quality. "
            elif feature in ["busy", "leisure"]:
                addition = f"It is a {feature} restaurant. "
            elif feature in ["short", "long"]:
                addition = f"It is a {feature} stay restaurant. "
            else:
                addition = f"It is a {feature} restaurant. "
        details = (f"According to your requirement. {addition}\n"
                   f"The food quality there is {data[7]}. The restaurant is {data[8]}. "
                   f"The time to stay in the restaurant is {data[9]}.\n"
                   f"Their phone number is {data[4]}. Their address is {data[5]}. Their postcode is {data[6]}.")
        info = self.capitalize(info)
        details = self.capitalize(details)
        print(info)
        if detail:
            print(details)
    def on_enter_welcome(self):
        print("Hello , welcome! You can ask for restaurants by area , price range or food type.")
        self.first_time = True
        self.status = "forward"

    def on_enter_ask_delay(self):
        print("Do you want a 3 second delay before showing system responses?")
        _, act = self.get_user_input()
        if act == 'affirm':
            self.delay = True
            self.status = "forward"
        elif act == 'negate':
            self.delay = False
            self.status = "forward"
        else:
            self.status = "again"

    def on_enter_ask_caps(self):
        print("Do you want the system responses are ALL CAPS?")
        _, act = self.get_user_input()
        if act == 'affirm':
            self.caps = True
            self.status = "forward"
        elif act == 'negate':
            self.caps = False
            self.status = "forward"
        else:
            self.status = "again"

    def on_enter_ask_area(self):
        print("What part of town do you have in mind?")
        message, act = self.get_user_input(area)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, area)
            if closest_key in area and min_distance < 3:
                self.status = "forward"
                self.filter['area'] = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"

    def on_enter_ask_price(self):
        print("What is your preferred price range? "
              "Would you like something in the cheap , moderate , or expensive price range?")
        message, act = self.get_user_input(price_range)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, price_range)
            if closest_key in price_range and min_distance < 3:
                self.status = "forward"
                self.filter['price_range'] = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"

    def on_enter_ask_food_type(self):
        print("What kind of food would you like?")
        message, act = self.get_user_input(food_type)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, food_type)
            if closest_key in food_type and min_distance < 3:
                self.status = "forward"
                self.filter['food_type'] = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"


    def on_enter_check_recommend(self):
        if self.first_time:
            self.recommend_list = search_restaurants(self.restaurants, self.filter)
            self.filter_list = self.recommend_list
            self.first_time = False
        # print(self.recommend_list)
        if len(self.recommend_list):
            self.status = "success"
        else:
            self.status = "failure"

    def on_enter_ask_addition(self):
        print("Do you have additional requirements? (enter no to show recommendation)")
        message, act = self.get_user_input()
        if act == 'negate':
            self.status = "skip"
            return
        closest_key, min_distance = find_nearest_keyword(message, additional)
        if closest_key in additional and min_distance < 3:
            self.status = "forward"
            self.filter['addition'].append(closest_key)
        else:
            self.status = "again"

    def on_enter_filter_recommend(self):
        self.filter_list = filter_restaurants(self.filter_list, self.filter)
        # print(self.filter_list)
        if len(self.filter_list):
            self.status = "success"
        else:
            self.status = "failure"

    def check_contradiction(self):
        if self.filter["food_type"] == "romanian" and "touristic" in self.filter["addition"]:
            print("It is impossible for a restaurant to provide romanian food and preferred by touristic at the same time!")
        if "long" in self.filter["addition"] and "children" in self.filter["addition"]:
            print("It is impossible for a restaurant to stay long and preferred by children at the same time!")
        if "busy" in self.filter["addition"] and "romantic" in self.filter["addition"]:
            print("It is impossible for a restaurant to be busy and be romantic at the same time!")

    def on_enter_ask_reset(self):
        self.check_contradiction()
        print("No recommendation exist! Do you want to reset the additional requirement?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "success"
            self.filter_list = self.recommend_list
            self.filter['addition'] = []
        else:
            self.status = "failure"

    def on_enter_give_recommend(self):
        print("We find a restaurant!")
        recommend = self.filter_list.iloc[0]
        self.print_restaurant(recommend)
        print("Do you want it as your final result?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "success"
        else:
            self.recommend_list = self.recommend_list.drop(recommend.name)
            self.status = "failure"

    def on_enter_ask_start_over(self):
        print("No recommendation exist! Do you want to start over?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "again"
        else:
            self.status = "forward"

    def on_enter_detail(self):
        self.print_restaurant(self.filter_list.iloc[0], True)
        self.status = "forward"

    def on_enter_end(self):
        print("Goodbye!")

def run1():
    root_path = "MAIR_23"
    # Set up classifier
    data_path = os.path.join(root_path, "data/dialog_acts.csv")
    # data_path_dedup = "../data/dialog_acts_dedup.csv"

    vectorizer = CountVectorizer()
    classifier = KNeighborsClassifier(3)

    # Choose here which dataset to use(normal or dedup)
    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
    classifier = train(vectorizer, classifier, X_train, y_train)

    # Load restaurant data
    restaurants = pd.read_csv(os.path.join(root_path, "data/restaurant_info_v2.csv"))

    # Create a door object
    system = Dialog(classifier, vectorizer, restaurants)
    while system.state != 'end':
        # print(system.state)
        # print(system.status)
        getattr(system, system.status)()

if __name__ == '__main__':
    run1()