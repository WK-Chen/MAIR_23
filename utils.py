import csv
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