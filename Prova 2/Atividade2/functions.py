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

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def plot_erros(maes, labels):
	y_pos = np.arange(len(maes))
	plt.bar(y_pos, maes, color=['red', 'blue', 'blue', 'blue', 'blue', 'purple'])
	plt.xticks(y_pos, labels, rotation='vertical')
	plt.ylabel('Mae')
	plt.title('Erros médios absolutos em cada caso')
	for i in range(len(maes)):
		plt.annotate("${:7.2f}".format(maes[i]), xy=(y_pos[i], maes[i]), ha='center', va='bottom')
	plt.show()


def train_and_plot(train_data, train_labels, test_data, test_labels, epochs, title_type = ''):
	# The patience parameter is the amount of epochs to check for improvement.
	model = build_model(train_data)
	model.summary()

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

	history = model.fit(train_data, train_labels, epochs=epochs,
	                    validation_split=0.2, verbose=0,
	                    callbacks=[early_stop, PrintDot()])

	[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

	print("\nErro médio absoluto: ${:7.2f}".format(mae * 1000))

	test_predictions = model.predict(test_data).flatten()

	plt.figure()
	plt.xlabel('Época')
	plt.ylabel('Erro médio absoluto')
	plt.title(f'Progresso de treinamento {title_type}')
	plt.plot(history.epoch, np.array(history.history['mae']),
	         label='Perca de treinamento')
	plt.plot(history.epoch, np.array(history.history['val_mae']),
	         label='Perca de Validação')
	plt.legend()
	plt.ylim([0, 5])
	plt.show()

	plt.scatter(test_labels, test_predictions)
	plt.xlabel('Valores verdadeiros [1000$]')
	plt.ylabel('Predições [1000$]')
	plt.title(f'Predições {title_type}')
	plt.axis('equal')
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())
	plt.plot([-100, 100], [-100, 100])
	plt.show()

	error = test_predictions - test_labels
	plt.hist(error, bins=50)
	plt.xlabel("Erro de predição [1000$]")
	plt.ylabel("Contagem")
	plt.title(f'Erros de predição {title_type}')
	plt.show()
	plt.pause(1)

	return mae * 1000