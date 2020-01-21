# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.preprocessing.importing_dataset import read_csv
from core.preprocessing.model_selection import train_test_split
from core.preprocessing.base import SimplePreprocessor

# Importing the dataset
dataset, x, y = read_csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3)
# processor = SimplePreprocessor('Salary_Data')
# processor.train(test_size=1/3)
# Fitting Single Linear Regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)


# Visualizing the training set resulst
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()