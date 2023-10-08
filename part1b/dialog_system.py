import os
import sys
sys.path.append(os.getcwd())

from transitions import Machine
from part1a.classifier1 import *
from utils.utils import *


# Keywords for prefer extraction
food_type = ["british", "modern european", "italian", "romanian", "chinese", "seafood", "steakhouse", "asian oriental",
             "french", "chinese", "portuguese", "indian", "spanish", "japanese", "european", "vietnamese", "korean",
             "thai", "moroccan", "swiss", "fusion", "gastropub", "tuscan", "international", "traditional",
             "mediterranean", "polynesian", "african", "turkish", "north american", "australasian", "persian",
             "jamaican", "lebanese", 'cuban', "catalan"]
area = ["east", "south", "centre", "north", "west"]
price_range = ["cheap", "moderate", "expensive"]


def prediction(classifier, vectorizer, utterance: str):
    res = predict(utterance, classifier, vectorizer)
    return res

def search_restaurants(data, filter):
    # lookup function to search restaurants from database
    res = data[
        (data['pricerange'] == filter['price_range']) & (data['area'] == filter['area']) & (
                    data['food'] == filter['food_type'])
        ]
    return res

class Dialog:
    def __init__(self, classifier, vectorizer, restaurants):
        # Define the states
        states = ['start', 'welcome', 'ask_area', 'area_confirm',
                  'ask_price', 'price_confirm', 'ask_food_type', 'food_confirm',
                  'check_recommend', 'give_recommend', 'ask_start_over',
                  'detail', 'end']

        # Define the transitions
        transitions = [
            {'trigger': 'forward', 'source': 'start', 'dest': 'welcome'},
            {'trigger': 'forward', 'source': 'welcome', 'dest': 'ask_area'},

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
            {'trigger': 'again', 'source': 'ask_food_type', 'dest': '='},

            {'trigger': 'forward', 'source': 'food_confirm', 'dest': 'check_recommend'},
            {'trigger': 'backward', 'source': 'food_confirm', 'dest': 'ask_food_type'},

            {'trigger': 'success', 'source': 'check_recommend', 'dest': 'give_recommend'},
            {'trigger': 'failure', 'source': 'check_recommend', 'dest': 'ask_start_over'},

            {'trigger': 'forward', 'source': 'ask_start_over', 'dest': 'end'},
            {'trigger': 'again', 'source': 'ask_start_over', 'dest': 'welcome'},

            {'trigger': 'success', 'source': 'give_recommend', 'dest': 'detail'},
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

        # Load data, Initialize filter, first search and recommend list
        self.restaurants = restaurants
        self.filter = {"area": "", "price_range": "", "food_type": ""}
        self.first_time = True
        self.recommend_list = []

    def get_user_input(self, field=None):
        message = input("User: ").lower()
        act = prediction(self.classifer, self.vectorizer, message)
        if act == "null" and field:
            closest_key, min_distance = find_nearest_keyword(message, field)
            act = prediction(self.classifer, self.vectorizer, closest_key)
        return message, act

    def print_restaurant(self, data, detail=False):
        print(f"I recommend {data[0]}, it is an {data[1]} {data[3]} restaurant in the {data[2]} of town.")
        if detail:
            print(f"Their phone number is {data[4]}. Their address is {data[5]}. Their postcode is {data[6]}.")
    def on_enter_welcome(self):
        print("Hello , welcome! You can ask for restaurants by area , price range or food type.")
        self.first_time = True
        self.status = "forward"
    def on_enter_ask_area(self):
        print("What part of town do you have in mind?")
        message, act = self.get_user_input(area)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, area)
            if closest_key in area and min_distance == 0:
                self.status = "forward"
                self.filter['area'] = closest_key
            elif closest_key in area and min_distance < 3:
                self.status = "confirm"
                self.confirm_text = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"

    def on_enter_area_confirm(self):
        print(f"I did not understand that correctly, do you mean {self.confirm_text} food?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "forward"
            self.filter['area'] = self.confirm_text
        else:
            self.status = "backward"

    def on_enter_ask_price(self):
        print("What is your preferred price range? "
              "Would you like something in the cheap , moderate , or expensive price range?")
        message, act = self.get_user_input(price_range)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, price_range)
            if closest_key in price_range and min_distance == 0:
                self.status = "forward"
                self.filter['price_range'] = closest_key
            elif closest_key in price_range and min_distance < 3:
                self.status = "confirm"
                self.confirm_text = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"

    def on_enter_price_confirm(self):
        print(f"I did not understand that correctly, do you mean to eat {self.confirm_text}?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "forward"
            self.filter['price_range'] = self.confirm_text
        else:
            self.status = "backward"

    def on_enter_ask_food_type(self):
        print("What kind of food would you like?")
        message, act = self.get_user_input(food_type)
        if act == 'inform':
            closest_key, min_distance = find_nearest_keyword(message, food_type)
            if closest_key in food_type and min_distance == 0:
                self.status = "forward"
                self.filter['food_type'] = closest_key
            elif closest_key in food_type and min_distance < 3:
                self.status = "confirm"
                self.confirm_text = closest_key
            else:
                self.status = "again"
        else:
            self.status = "again"

    def on_enter_food_confirm(self):
        print(f"I did not understand that correctly, do you mean {self.confirm_text} food?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "forward"
            self.filter['food_type'] = self.confirm_text
        else:
            self.status = "backward"

    def on_enter_check_recommend(self):
        if self.first_time:
            res = search_restaurants(self.restaurants, self.filter)
            self.recommend_list = res.values.tolist()
            self.first_time = False
        if self.recommend_list:
            self.status = "success"
        else:
            self.status = "failure"

    def on_enter_give_recommend(self):
        print("We find a restaurant!")
        self.print_restaurant(self.recommend_list[0])
        print("Do you want it as your final result?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "success"
        else:
            self.recommend_list.pop(0)
            self.status = "failure"

    def on_enter_ask_start_over(self):
        print("No recommendation exist! Do you want to start over?")
        message, act = self.get_user_input()
        if act == 'affirm':
            self.status = "again"
        else:
            self.status = "forward"

    def on_enter_detail(self):
        self.print_restaurant(self.recommend_list[0], True)
        self.status = "forward"

    def on_enter_end(self):
        print("Goodbye!")


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Set up classifier
    data_path = os.path.join(root_path, "data/dialog_acts.csv")
    # data_path_dedup = "../data/dialog_acts_dedup.csv"

    vectorizer = CountVectorizer()
    classifier = KNeighborsClassifier(3)

    # Choose here which dataset to use(normal or dedup)
    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
    classifier = train(vectorizer, classifier, X_train, y_train)

    # Load restaurant data
    restaurants = pd.read_csv(os.path.join(root_path, "data/restaurant_info.csv"))

    # Create a door object
    system = Dialog(classifier, vectorizer, restaurants)
    while system.state != 'end':
        getattr(system, system.status)()