# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:27:13 2020

Keras for Beginners: Building Your First Neural Network
A beginner-friendly guide on using Keras to implement a simple
Neural Network in Python

The Problem: MNIST digit classification

https://victorzhou.com/blog/keras-neural-network-tutorial/

@author: Rafael
"""

import numpy as np
import mnist
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import to_categorical # transforma classes de inteiro para binario
#from keras.utils import plot_model

import matplotlib.pyplot as plt

Nepochs = 5     # numero de epocas/steps/iterations
Nbatch  = 32    # numero de padroes no treinamento modo mini-batch

# A primeira vez que vc rodar isto pode ser um pouco demorado, uma vez que
# o modulo mnist precisa ser carregado.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
print('Treinamento:{}'.format(train_images.shape)) # (60000, 28, 28)
print('Labels treinamento:{}'.format(train_labels.shape)) # (60000,)
print('Teste:{}'.format(test_images.shape)) # (60000, 28, 28)
print('Labels testes:{}'.format(test_labels.shape)) # (60000,)

Ntrain = train_images.shape[0]  # num. padroes de treinamento
Ntest = test_images.shape[0]    # num. padroes de teste

# visualizacao de alguns digitos e seus labels
for i in range(3):
    plt.figure()
    plt.title('Label is {label}'.format(label=train_labels[i]))
    plt.imshow(train_images[i,::], cmap='gray')
    plt.show()

#%% Normalizamos as imagens
train_images = (train_images / 255)
test_images = (test_images / 255)
# Achatar as imagens em um vetor (padroes de entrada vetoriais)
rows = train_images.shape[1]
cols = train_images.shape[2]
M = rows*cols # 784
# Reshape para 1D (padroes na forma vetorial)
train_images = train_images.reshape((-1, M))
test_images = test_images.reshape((-1, M))

#%% Construindo o modelo
''' O construtor Sequential monta as camadas, neste exemplo,
 camadas densas (Dense layer, completamente conectada).
 Vamos usar 3 camadas: 2 intermediárias com 64 neuronios e
 funcao de ativacao sigmoide tanh (ou relu).
 Precisamos sempre especificar a dimensao da entrada da rede
 atraves de input_shape na primeira camada no modelo Sequential.
'''
model = Sequential([
        Dense(32, activation='tanh', input_shape=(M,)),
        Dense(32, activation='tanh'),
        Dense(10, activation='softmax'),
        ])

#%% Configurando o treinamento
'''
Decidimos por 3 fatores: 1)optimizer; 2)loss function;
        3) lista de metricas:
Opcoes:
    1) optimizer: Adam, SGD, Adadelta, Adagrad, RMSprop
    2) loss: (classificacao): categorical_crossentropy (>2 classes)
                        (regressao): MeanSquaredError, MeanAbsoluteError
    3) metrics: accuracy
'''
model.compile(optimizer=optimizers.Adam(),        #Adam(lr=0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

# imprime um resumo da rede (modelo)
model.summary()

#%% treinando o modelo (invocar a funcao fit())
history = model.fit(train_images,
          to_categorical(train_labels),
          epochs=5,
          batch_size=32,
          shuffle=True,
          validation_split=1/3,
          #validation_data=(test_images, to_categorical(test_labels)),
          )
'''
Aqui, a funcao to_categorical() codifica um vetor de inteiros (numpy array)
    das classes para valores binarios (0s e 1s) e resulta em
    um numpy array (na forma de matriz) com numero de colunas
    igual ao num. de classes/categorias dos dados.
'''
# exemplo de to_categorical (descomente as duas linhas seguintes)
#train_labels_encoded = to_categorical(np.arange(0,10))
#print(train_labels_encoded)
#%% testando o modelo
model.evaluate(test_images,
               to_categorical(test_labels),
               )
# salvando os pesos do modelo treinado e o modelo inteiro
model.save_weights('model_weights.h5')  # pesos
model.save('model.h5')                  # modelo inteiro
'''
A funcao load_weights() a seguir carrega o modelo treinado (uso futuro):
    model.load_weights('model.h5') # carrega o modelo salvo
'''
#%% usando o modelo para predicao da saida (processar os dados e ter uma decisao)
# Processa as 5 primeiras imagens de teste
predictions = model.predict(test_images[:5])
print('\nsaida de predict():    {}'.format(np.argmax(predictions, axis=1))) 
print('labels correspondentes:{}'.format(test_labels[:5]))

#%% imprimindo resultados do treinamento
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Época x Precisão')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.margins(x=0)
plt.legend(['Treinamento', 'Validação'], loc='upper left')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Época x Erro')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.margins(x=0)