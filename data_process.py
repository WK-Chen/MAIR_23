from utils import *


def dat_to_csv(input_path, output_path):
    data = []
    # Read the input file and split lines into tag and sentence
    with open(input_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                tag, sentence = parts
                data.append([tag.lower(), sentence.lower()])
    save_csv(output_path, data)


def deduplicate(input, output):
    # Create a new .csv file for dataset without duplication
    data = load_csv(input)
    data_dedup = []
    for sample in data:
        if sample not in data_dedup:
            data_dedup.append(sample)
    save_csv(output, data_dedup)


# We may change .csv format to .jsonl format later

if __name__ == '__main__':
    data_path = './data/dialog_acts.dat'
    # dat_to_csv(data_path, "./data/dialog_acts.csv")
    # deduplicate(data_path, "./data/dialog_acts_dedup.csv")
    data = load_csv("./data/dialog_acts.csv")
    print(data[0])