import pandas as pd
from utils import *

data_path = "./data/dialog_acts.csv"
data_path_dedup = "./data/dialog_acts_dedup.csv"

df = pd.DataFrame(load_csv(data_path), columns=["Label", "Sentence"])

print(df)
