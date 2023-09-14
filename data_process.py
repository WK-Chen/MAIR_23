from utils import *
import random


# Reads the given dataset and converts it to a csv file
def dat_to_csv(input_path, output_path):
    data = []
    with open(input_path, "r") as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                tag, sentence = parts
                data.append([tag.lower(), sentence.lower()])
    random.shuffle(data)
    save_csv(output_path, data)


# Given the data converted to csv, it removes all duplicates and saves it again as a csv
def deduplicate(input, output):
    data = load_csv(input)
    data_dedup = []
    for sample in data:
        if sample not in data_dedup:
            data_dedup.append(sample)
    random.shuffle(data_dedup)
    save_csv(output, data_dedup)


# We may change .csv format to .jsonl format later

if __name__ == "__main__":
    # Define Paths
    orig_data_path = "./data/dialog_acts.dat"
    data_path = "./data/dialog_acts.csv"
    data_path_dedup = "./data/dialog_acts_dedup.csv"

    # Create the csv datasets
    # dat_to_csv(orig_data_path, data_path)
    # deduplicate(data_path, data_path_dedup)

    # Load the datasets
    data = load_csv("./data/dialog_acts.csv")
    data_dedup = load_csv("./data/dialog_acts_dedup.csv")
    print(len(data), len(data_dedup))
