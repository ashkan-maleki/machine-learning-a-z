# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from core.preprocessing.importing_dataset import read_csv
from core.preprocessing.model_selection import train_test_split

def impute(X):
    from sklearn.impute import SimpleImputer as Imputer
    imputer = Imputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X


def encode(x, y):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    label_encode_x = LabelEncoder()
    x[:, 0] = label_encode_x.fit_transform(x[:, 0])

    from sklearn.compose import ColumnTransformer

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )

    x = np.array(ct.fit_transform(x), dtype=np.int)

    label_encode_y = LabelEncoder()
    y = label_encode_y.fit_transform(y)
    return x, y




def scalar(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    return sc_x.fit_transform(x_train),\
        sc_x.transform(x_test)


# Importing the dataset

dataset, x, y = read_csv('Data.csv')
x = impute(x)
x, y = encode(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train, x_test = scalar(x_train, x_test)

