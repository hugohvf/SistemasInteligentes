# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:50:37 2020

CNN basica com Keras para classificar as imagens
dos digitos escritos a mao da base MNIST.

@author: Rafael
"""

import numpy as np
import mnist
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.utils import to_categorical # transforma classes de inteiro para binario
#from keras.utils import plot_model

import matplotlib.pyplot as plt

Nepochs = 5     # numero de epocas/steps/iterations
Nbatch  = 32    # numero de padroes no treinamento modo mini-batch
Nclasses = 10   # numero de classes

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
#train_images = train_images.reshape((-1, M))
#test_images = test_images.reshape((-1, M))
#input_shape = (1, rows, cols)

# Reshape para padroes 2D na forma (rows,cols,1): 1 canal pois eh grayscale
train_images = train_images.reshape(train_images.shape[0], rows, cols, 1)
test_images = test_images.reshape(test_images.shape[0], rows, cols, 1)
input_shape = (rows, cols, 1)

#%% Construindo o modelo
''' O construtor Sequential monta as camadas, neste exemplo,
 camadas convolucionais (com pooling e dropout) e  2 densas
 (Dense layer, completamente conectada).
 Vamos usar 4 camadas: 2 convolucionais seguidas de 2 densas
 com funcoes de ativacao relu e softmax na ultima camada.
 Precisamos sempre especificar a dimensao da entrada da rede
 atraves de input_shape na primeira camada no modelo Sequential.
'''

model = Sequential()
# camada de entrada: convolucional
model.add(Conv2D(32, # numero de filtros (depth)
                 kernel_size=(3, 3), # tamanho dos filtros 
                 input_shape=input_shape,
                 activation='relu',
                 padding='same', # entrada e saida tem mesmas dimensoes
                 strides=(2,2),)) # passo duplo
model.add(MaxPooling2D(pool_size=(2, 2))) # subamostragem  pelo filtro de maximo, reduz o numero de parametros a serem treinados
model.add(Dropout(0.25)) # desativa alguns neuronios durante
                        # o treinamento para tornar a rede mais generalista (evitar overfitting)
# 2a camada: convolucional
model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # 1D (saimos do 2D para 1D)
# 3a camada: densa
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(Nclasses, activation='softmax'))
    
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