"""
The dataset contains 13 different features:
	Per capita crime rate.
	The proportion of residential land zoned for lots over 25,000 square feet.
	The proportion of non-retail business acres per town.
	Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
	Nitric oxides concentration (parts per 10 million).
	The average number of rooms per dwelling.
	The proportion of owner-occupied units built before 1940.
	Weighted distances to five Boston employment centers.
	Index of accessibility to radial highways.
	Full-value property-tax rate per $10,000.
	Pupil-teacher ratio by town.
	1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
	Percentage lower status of the population.
	@author:
"""

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as f

boston_housing = keras.datasets.boston_housing
# Usar este caso não tenha baixado os dados
# boston_housing = tf.keras.datasets.boston_housing.load_data(
#     path="boston_housing.npz", test_split=0.2, seed=113
# )

EPOCHS = 500

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)

print(df.head())

# Test data is *not* used when calculating the mean and std.
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print('Usando todos')
f.train_and_plot(train_data, train_labels, test_data, test_labels, EPOCHS)

print('Removendo 3 primeiros')
f.train_and_plot(train_data[:, 2:-1], train_labels, test_data[:, 2:-1], test_labels, EPOCHS)

print('Removendo 3 ultimos')
f.train_and_plot(train_data[:, 0:-3], train_labels, test_data[:, 0:-3], test_labels, EPOCHS)

print('Removendo 6 primeiros')
f.train_and_plot(train_data[:, 5:-1], train_labels, test_data[:, 5:-1], test_labels, EPOCHS)

print('Removendo 6 ultimos')
f.train_and_plot(train_data[:, 0:-6], train_labels, test_data[:, 0:-6], test_labels, EPOCHS)

print('Selecionando 3')
f.train_and_plot(np.column_stack((train_data[:, 7], train_data[:, 12], train_data[:, 11])), train_labels,
                 np.column_stack((test_data[:, 7], test_data[:, 12], test_data[:, 11])), test_labels, EPOCHS)