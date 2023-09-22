from classifier1 import *
import Levenshtein
import random
import pandas as pd

# Keywords for prefer extraction
food_type = ["british", "modern european", "italian", "romanian", "chinese", "seafood", "steakhouse", "asian oriental",
             "french", "chinese", "portuguese", "indian", "spanish", "japanese", "european", "vietnamese", "korean",
             "thai", "moroccan", "swiss", "fusion", "gastropub", "tuscan", "international", "traditional",
             "mediterranean", "polynesian", "african", "turkish", "north american", "australasian", "persian",
             "jamaican", "lebanese", 'cuban', "catalan"]
area = ["east", "south", "centre", "north", "west"]
price_range = ["cheap", "moderate", "expensive"]

categories = [food_type, area, price_range]

# Define dialog states
DIALOG_STATE_INITIAL = 0
DIALOG_STATE_CUISINE = 1
DIALOG_STATE_CUISINE_CONFIRMATION = 2
DIALOG_STATE_LOCATION = 3
DIALOG_STATE_LOCATION_CONFIRMATION = 4
DIALOG_STATE_PRICE_RANGE = 5
DIALOG_STATE_PRICE_RANGE_CONFIRMATION = 6
DIALOG_STATE_GOT_PREFERENCES = 7
# Fill in all other required ones
DIALOG_END = -1

# This code still does not account for 'any' (non)preference
# INITIALIZATION
dialog_state = {
    "area": None,
    "price_range": None,
    "food_type": None,
}


def prediction(utterance: str):
    res = predict(utterance, classifier, vectorizer)
    print("label:", res)
    return res


# Define system responses
def get_sentence(key, f_thing=None):
    system_responses = {
        "error": "I'm sorry, I didn't understand. Please provide your preferences for food type, location, and price range.",
        "welcome_message": "Hello , welcome! You can ask for restaurants by area , price range or food type. How can I help?",
        "ask_food_type": "What kind of food would you like?",
        "ask_area": "What part of town do you have in mind?",
        "ask_price_range": "What is your preferred price range? Would you like something in the cheap , moderate , or expensive price range?",
        "confirm_food_type": f"I did not understand that correctly, do you mean {f_thing} food?",
        "confirm_area": f"I did not understand that correctly, do you mean in the {f_thing} of town?",
        "confirm_price_range": f"I did not understand that correctly, do you mean to eat {f_thing}?",
        # Below still needs to give an actual recommendation
        "recommend": f"Restaurant {f_thing} would be a great option!",
        # Fill in all other responses
        "goodbye": "See ya!"
    }
    return system_responses[key]


def var_name(var):
    # Create a dictionary with the variable as the value and its name as the key
    variables = globals()
    for name, value in variables.items():
        if value is var:
            return name
    return None  # Return None if the variable is not found


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


# The state_transition function managing the conversation
def state_transition(dialog_state_n: int, user_utterance: str):
    # Define the users act
    user_utterance = user_utterance.lower()
    closest_key = user_utterance
    act = prediction(user_utterance)
    if act == "null":
        closest_key, min_distance = find_nearest_keyword(user_utterance, food_type + area + price_range)
        act = prediction(closest_key)

    # If in initial false or requesting additional data
    if dialog_state_n == DIALOG_STATE_INITIAL:
        # If the user is informing
        if act == "inform":
            # Check for each category, if a keyword is given
            closest_key, min_distance = find_nearest_keyword(user_utterance, food_type + area + price_range)
            confirm = False
            for el in categories:
                if closest_key in el and dialog_state[var_name(el)] is None:
                    if min_distance == 0:
                        dialog_state[var_name(el)] = closest_key
                    elif min_distance < 3:
                        confirm = var_name(el)
                        word_to_confirm = closest_key
                        dialog_state[var_name(el)] = closest_key

            # If needed to confirm, go to that confirmation state
            if confirm:
                if confirm == "food_type":
                    return DIALOG_STATE_CUISINE_CONFIRMATION, get_sentence("confirm_food_type", word_to_confirm)
                elif confirm == "price_range":
                    return DIALOG_STATE_PRICE_RANGE_CONFIRMATION, get_sentence("confirm_price_range", word_to_confirm)
                elif confirm == "area":
                    return DIALOG_STATE_LOCATION_CONFIRMATION, get_sentence("confirm_area", word_to_confirm)
    elif dialog_state_n == DIALOG_STATE_CUISINE:
        closest_key, min_distance = find_nearest_keyword(user_utterance, food_type)
        if act == "inform":
            # Check for each category, if a keyword is given
            confirm = False
            if closest_key in food_type and dialog_state["food_type"] is None:
                if min_distance == 0:
                    dialog_state["food_type"] = closest_key
                elif min_distance < 3:
                    confirm = True
                    word_to_confirm = closest_key
                    dialog_state["food_type"] = closest_key

            # If needed to confirm, go to that confirmation state
            if confirm:
                return DIALOG_STATE_CUISINE_CONFIRMATION, get_sentence("confirm_food_type", word_to_confirm)
    elif dialog_state_n == DIALOG_STATE_LOCATION:
        closest_key, min_distance = find_nearest_keyword(user_utterance, area)
        if act == "inform":
            # Check for each category, if a keyword is given
            confirm = False
            if closest_key in area and dialog_state["area"] is None:
                if min_distance == 0:
                    dialog_state["area"] = closest_key
                elif min_distance < 3:
                    confirm = True
                    word_to_confirm = closest_key
                    dialog_state["area"] = closest_key

            # If needed to confirm, go to that confirmation state
            if confirm:
                return DIALOG_STATE_LOCATION_CONFIRMATION, get_sentence("confirm_area", word_to_confirm)
    elif dialog_state_n == DIALOG_STATE_PRICE_RANGE:
        closest_key, min_distance = find_nearest_keyword(user_utterance, price_range)
        if act == "inform":
            # Check for each category, if a keyword is given
            confirm = False
            if closest_key in price_range and dialog_state["price_range"] is None:
                if min_distance == 0:
                    dialog_state["price_range"] = closest_key
                elif min_distance < 3:
                    confirm = True
                    word_to_confirm = closest_key
                    dialog_state["price_range"] = closest_key

            # If needed to confirm, go to that confirmation state
            if confirm:
                return DIALOG_STATE_PRICE_RANGE_CONFIRMATION, get_sentence("confirm_price_range", word_to_confirm)
    # If instead in a confirmation fase
    elif dialog_state_n in [DIALOG_STATE_CUISINE_CONFIRMATION, DIALOG_STATE_LOCATION_CONFIRMATION,
                            DIALOG_STATE_PRICE_RANGE_CONFIRMATION]:
        # Possibly confirm
        if act == "affirm":
            ...
        elif act == "negate":
            if dialog_state_n == DIALOG_STATE_CUISINE_CONFIRMATION:
                dialog_state["food_type"] = None
            elif dialog_state_n == DIALOG_STATE_PRICE_RANGE_CONFIRMATION:
                dialog_state["price_range"] = None
            elif dialog_state_n == DIALOG_STATE_LOCATION_CONFIRMATION:
                dialog_state["area"] = None

    # If not all information is known, ask for information that is not yet achieved
    if dialog_state_n < DIALOG_STATE_GOT_PREFERENCES:
        # Ask for the missing information
        if dialog_state["area"] == None:
            return DIALOG_STATE_LOCATION, get_sentence("ask_area")
        elif dialog_state["price_range"] == None:
            return DIALOG_STATE_PRICE_RANGE, get_sentence("ask_price_range")
        elif dialog_state["food_type"] == None:
            return DIALOG_STATE_CUISINE, get_sentence("ask_food_type")
        # Unless that information is now present! Go advise a restaurant or say its not possible
        else:
            # Still needs to pick a restaurant or tell there is no option available
            fit_restaurant = None
            return DIALOG_STATE_GOT_PREFERENCES, get_sentence("recommend", dialog_state)

    # Go over the other states after getting the preferences
    elif dialog_state_n == DIALOG_STATE_GOT_PREFERENCES:
        return DIALOG_END, None


def search_restaurants(data, filter):
    res = data[
        (data['pricerange'] == filter['price_range']) & (data['area'] == filter['area']) & (
                    data['food'] == filter['food_type'])
        ]
    return res


def dialog_system():
    current_state = DIALOG_STATE_INITIAL
    print("System: ", get_sentence("welcome_message"))
    while current_state != DIALOG_END:
        user_input = input("User: ").lower()
        current_state, system_response = state_transition(current_state, user_input)
        print("System: ", system_response)
    return dialog_state


if __name__ == '__main__':
    # Set up classifier
    data_path = "./data/dialog_acts.csv"
    data_path_dedup = "./data/dialog_acts_dedup.csv"

    vectorizer = CountVectorizer()
    classifier = KNeighborsClassifier(3)

    # Choose here which dataset to use(normal or dedup)
    X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
    classifier = train(vectorizer, classifier, X_train, y_train)

    # Load restaurant data
    restaurants = pd.read_csv("data/restaurant_info.csv")

    # The system
    pereference = dialog_system()
    res = search_restaurants(restaurants, pereference)
    print("I found a restaurant for you!")
    print(res)
