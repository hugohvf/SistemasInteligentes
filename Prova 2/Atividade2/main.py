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

	@author: Ana Beatriz, Caio Caetano, Hugo Fusinato, Lucas Barzotto
"""

from tensorflow import keras
import numpy as np
import functions as f

boston_housing = keras.datasets.boston_housing
# boston_housing = tf.keras.datasets.boston_housing.load_data(
#     path="boston_housing.npz", test_split=0.2, seed=113
# )

epochs = 500

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

# Test data is *not* used when calculating the mean and std.
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Erro médio absoluto
maes = []

print('Treinando usando todos os parâmetros')
maes.append(f.train_and_plot(train_data, train_labels, test_data, test_labels, epochs, 'para todos os parâmetros'))

print('Removendo 3 primeiros parâmetros')
maes.append(f.train_and_plot(train_data[:, 2:-1], train_labels, test_data[:, 2:-1], test_labels, epochs,
                             'sem os três primeiros parâmetros'))

print('Removendo 3 ultimos parâmetros')
maes.append(f.train_and_plot(train_data[:, 0:-3], train_labels, test_data[:, 0:-3], test_labels, epochs,
                             'sem os três últimos parâmetros'))

print('Removendo 6 primeiros parâmetros')
maes.append(f.train_and_plot(train_data[:, 5:-1], train_labels, test_data[:, 5:-1], test_labels, epochs,
                             'sem os seis primeiros parâmetros'))

print('Removendo 6 ultimos parâmetros')
maes.append(f.train_and_plot(train_data[:, 0:-6], train_labels, test_data[:, 0:-6], test_labels, epochs,
                             'sem os seis últimos parâmetros'))

print('Selecionando 3 parâmetros')
selected_params = ['CRIM', 'LSTAT', 'RM']

selected_test_data = selected_train_data = []
for param in selected_params:
	if param in column_names:
		index = column_names.index(param)

		if param == selected_params[0]:
			selected_train_data = train_data[:, index]
			selected_test_data = test_data[:, index]
		else:
			selected_train_data = np.column_stack((selected_train_data, train_data[:, index]))
			selected_test_data = np.column_stack((selected_test_data, test_data[:, index]))

		print(selected_train_data.shape, selected_test_data.shape)
	else:
		raise Exception("Parametro com nome errado")

maes.append(f.train_and_plot(selected_train_data, train_labels, selected_test_data, test_labels, epochs,
                             'com ' + ' '.join(selected_params)))

labels = ['Todos', 'Sem 3 primeiros', 'Sem 3 últimos', 'Sem 6 primeiros', 'Sem 6 ultimos', ' '.join(selected_params)]

f.plot_erros(maes, labels)
