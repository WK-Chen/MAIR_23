from classifier1 import *
import Levenshtein
import random


# Set up classiffier
data_path = "./data/dialog_acts.csv"
data_path_dedup = "./data/dialog_acts_dedup.csv"

vectorizer = CountVectorizer()
classifier = KNeighborsClassifier(3)

# Choose here which dataset to use (normal or dedup)
X_train, X_test, y_train, y_test = split_dataset_pd(data_path)
classifier = train(vectorizer, classifier, X_train, y_train)

def prediction(utterance: str):
    return predict(utterance, classifier, vectorizer)

# Define system responses
system_responses = {
    "welcome_message": "Hello , welcome! You can ask for restaurants by area , price range or food type.",
    "ask_food_type": "What kind of food would you like?",
    "ask_area": "What part of town do you have in mind?",
    "ask_price_range": "What is your preferred price range? Would you like something in the cheap , moderate , or expensive price range?",
    "preferences_extracted": "Great! Here is a restaurant that matches your preferences: {restaurant}",
    "preferences_not_extracted":"Unfortunately we could not find any restaurants matching. Would you like to start over?",
    "ask_confirmation":"",
    "error": "I'm sorry, I didn't understand. Please provide your preferences for food type, location, and price range.",
}

# Keywords for prefer extraction
food_types = ["british","modern european","italian", "romanian", "chinese", "seafood","steakhouse", "asian oriental", "french","chinese","portuguese", "indian","spanish", "japanese", "european", "vietnamese", "korean", "thai", "moroccan", "swiss", "fusion", "gastropub", "tuscan", "international", "traditional", "mediterranean", "polynesian", "african", "turkish", "north american", "australasian", "persian", "jamaican", "lebanese", 'cuban', "catalan"]
areas = ["east", "south", "centre", "north", "west"]
price_ranges = ["cheap", "moderate", "expensive"]



# Define initial dialog state
dialog_state = {
    "area": None,
    "price_range": None,
    "food_type": None,
}

# The state_transition function managing the conversation
def state_transition(dialog_state, user_utterance):
    