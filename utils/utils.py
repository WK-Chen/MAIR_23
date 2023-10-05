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