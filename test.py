import pandas as pd
from utils import *

data_path = "./data/dialog_acts.csv"
data_path_dedup = "./data/dialog_acts_dedup.csv"

data = load_csv(data_path)
print(data[:5])
assert False

df = pd.DataFrame(load_csv(data_path), columns=["Label", "Sentence"])

print(df)

if __name__ == "__main__":
    (x_train, y_train), (_,_) = keras.datasets.mnist.load_data()
    x_train = x_train / 255
    y_train = keras.utils.to_categorical(y_train, 10)

    sample_x = x_train[0]
    sample_y = y_train[0]

    k_horizontal = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])

    k_vertical = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])

    res1 = my_conv2d(sample_x, k_horizontal)
    res2 = my_conv2d(sample_x, k_vertical)
    print(res1.shape)
    print(res2.shape)