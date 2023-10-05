import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import random

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Load the CSV file into a DataFrame
    input_file = os.path.join(root_path, 'data/restaurant_info.csv')
    output_file = os.path.join(root_path, 'data/restaurant_info_v2.csv')
    df = pd.read_csv(input_file)

    # (food quality, crowdedness, length of stay)
    # Add three new properties to each line

    df['food_quality'] = [random.choice(['good', 'bad']) for _ in range(len(df))]
    df['crowdedness'] = [random.choice(['busy', 'leisure']) for _ in range(len(df))]
    df['length_of_stay'] = [random.choice(['long', 'short']) for _ in range(len(df))]

    print(df.head(5))

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print("New CSV file saved as", output_file)
