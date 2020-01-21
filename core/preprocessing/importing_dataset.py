import pandas as pd


def read_csv(file_name):
    dataset = pd.read_csv(file_name)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return dataset, x, y


