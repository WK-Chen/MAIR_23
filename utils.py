import csv
import pandas as pd

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