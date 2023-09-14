import csv


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
