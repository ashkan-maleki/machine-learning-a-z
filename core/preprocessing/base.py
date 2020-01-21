import pandas as pd

from sklearn.model_selection import train_test_split


class BasePreprocessor():
    def __init__(self, file_name):
        self._dataset = pd.read_csv(file_name)
        self._x = self._dataset.iloc[:, :-1].values
        self._y = self._dataset.iloc[:, -1].values

    def get_dataset(self):
        return self._dataset

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def train(self, test_size=0.2):
        self._x_train, self._x_test, self._y_train, self._y_test \
            = train_test_split(self._x, self._y, test_size=test_size)

    def get_x_train(self):
        return self._x_train

    def get_x_test(self):
        return self._x_test

    def get_y_train(self):
        return self._y_train

    def get_y_test(self):
        return self._y_train


class SimplePreprocessor(BasePreprocessor):
    pass