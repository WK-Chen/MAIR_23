import csv


def dat_to_csv(input_path, output_path):
    data = []
    # Read the input file and split lines into tag and sentence
    with open(input_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                tag, sentence = parts
                data.append([tag, sentence])
    save_csv(output_path, data)


def load_csv(path: str):
    data = []
    # Open the CSV file and read its contents
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader]
    return data


def save_csv(path: str, data):
    # Write the data to a CSV file
    with open(path, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)

    print(f"Conversion complete. Data saved in '{path}'.")


def deduplicate(input, output):
    # Create a new .csv file for dataset without duplication
    data = load_csv(input)
    data_dedup = []
    for sample in data:
        if sample not in data_dedup:
            data_dedup.append(sample)
    save_csv(output, data_dedup)


if __name__ == '__main__':
    data_path = './data/dialog_acts.dat'
    # dat_to_csv(data_path, "./data/dialog_acts.csv")
    # deduplicate(data_path, "./data/dialog_acts_dedup.csv")
