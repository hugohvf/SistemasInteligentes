import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def build_model(train_data):
	model = keras.Sequential([
		keras.layers.Dense(64, activation=tf.nn.relu,
		                   input_shape=(train_data.shape[1],)),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(1)
	])

	optimizer = tf.compat.v1.train.RMSPropOptimizer(0.001)

	model.compile(loss='mse',
	              optimizer=optimizer,
	              metrics=['mae'])
	return model


def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(history.epoch, np.array(history.history['mae']),
	         label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mae']),
	         label='Val loss')
	plt.legend()
	plt.ylim([0, 5])
	plt.show()


# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')


def train_and_plot(train_data, train_labels, test_data, test_labels, EPOCHS):
	# The patience parameter is the amount of epochs to check for improvement.
	model = build_model(train_data)
	model.summary()

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

	history = model.fit(train_data, train_labels, epochs=EPOCHS,
	                    validation_split=0.2, verbose=0,
	                    callbacks=[early_stop, PrintDot()])

	plot_history(history)

	[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

	print("\nTesting set Mean Abs Error: ${:7.2f}".format(mae * 1000))

	test_predictions = model.predict(test_data).flatten()

	plt.scatter(test_labels, test_predictions)
	plt.xlabel('True Values [1000$]')
	plt.ylabel('Predictions [1000$]')
	plt.axis('equal')
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())
	_ = plt.plot([-100, 100], [-100, 100])
	plt.show()

	error = test_predictions - test_labels
	plt.hist(error, bins=50)
	plt.xlabel("Prediction Error [1000$]")
	_ = plt.ylabel("Count")
	plt.show()
