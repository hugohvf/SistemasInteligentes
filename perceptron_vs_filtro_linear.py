# -*- coding: utf-8 -*-
"""
Perceptron vs filtro linear utilizando LMS
Problema de equalizacao de canal
Created on Mon Nov  2 16:14:19 2020

@author: 
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import lfilter
import cria_Xn_equalizacao
from scipy import *

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

M = int(3)  # dimensao de dados de entrada (portanto do filtro tambem)
N = int(1e3)   # num. de sinais transmitidos (num. de padroes) no aprendizado
mu = 0.01

sigma2_v = 0.05   # variancia do ruido
h_canal = [1., .4]    # h = 1 + 0.4*z-1

# geracao dos dados
sn_N = np.sign(np.random.randn(N, 1))
vn_N = math.sqrt(sigma2_v)*(np.random.randn(N, 1))
sch_N = lfilter(h_canal, 1, sn_N, axis=0)
xn_N = sch_N + vn_N

# organiza os dados na matriz Xn [M]:[N]
Xn = cria_Xn_equalizacao.equaliza(xn_N, N, M)
Xn_p = np.vstack((Xn, np.ones((1, N))))  # ones eh o bias (que nao tem ruido)

# pre-alocacao de memoria e inicializacao
e2_filtro = np.zeros((N, 1))
e2_perceptron = np.zeros((N, 1))
Wn_filtro = np.zeros((M, N+1))
Wn_perceptron = np.zeros((M+1, N+1))
Yn_filtro = np.zeros((N, 1))
Yn_perceptron = np.zeros((N, 1))

# treinamento do perceptron e do neuronio linear
for n in range(0, N):
    dn = sn_N[n, 0]
    xn = Xn[:, n]
    xn_p = Xn_p[:, n]

    # filtro linear
    wn = Wn_filtro[:, n]
    yn = np.matrix(wn) * np.matrix(xn).T
    erro = dn - yn
    Wn_filtro[:, n+1] = wn + mu * erro * xn
    e2_filtro[n, 0] = np.power(erro, 2)
    Yn_filtro[n, 0] = yn

    # perceptron
    wn = Wn_perceptron[:, n]
    yn = np.sign(np.matrix(wn) * np.matrix(xn_p).T)
    erro = dn - yn
    Wn_perceptron[:, n+1] = wn + mu*erro*xn_p
    e2_perceptron[n, 0] = np.power(erro, 2)
    Yn_perceptron[n, 0] = yn

##
vetor_n = np.array(np.arange(1, N + 1))

neuronio_linear, = plt.plot(vetor_n, e2_filtro, 'b', label="neuronio linear")
perceptron, = plt.plot(vetor_n, e2_perceptron, 'r', label="perceptron")
plt.title("Curvas de erro quadrático", fontdict=font)
plt.legend(handles=[neuronio_linear, perceptron])
plt.show()

ch_1, = plt.plot(vetor_n, sch_N, 'o')
plt.xlabel("n", fontdict=font)
plt.ylabel("channel output", fontdict=font)
plt.title("channel output (without noise)", fontdict=font)
plt.legend(handles=[ch_1])
plt.show()

ch_2, = plt.plot(vetor_n, xn_N, '.')
plt.xlabel("n", fontdict=font)
plt.ylabel("channel output", fontdict=font)
plt.title("channel output (with noise)", fontdict=font)
plt.legend(handles=[ch_2])
plt.show()

linear_filter, = plt.plot(vetor_n, Yn_filtro, '.', label="Linear filter")
perceptron, = plt.plot(vetor_n, Yn_perceptron, '.', label="Perpeptron")
plt.xlabel("n", fontdict=font)
plt.ylabel("channel output", fontdict=font)
plt.title("Equalizer output", fontdict=font)
plt.legend(handles=[linear_filter, perceptron])
plt.show()

for i in range(0, M):
    plt.plot(vetor_n, Wn_filtro[i, 0:-1])
plt.xlabel("n", fontdict=font)
plt.ylabel("pesos", fontdict=font)
plt.title("Evolucao dos pesos do filtro", fontdict=font)
plt.legend(handles=[linear_filter, perceptron])
plt.show()


for i in range(0, M):
    plt.plot(vetor_n, Wn_perceptron[i, 0:-1])
plt.xlabel("n", fontdict=font)
plt.ylabel("pesos", fontdict=font)
plt.title("Evolucao dos pesos do perceptron", fontdict=font)
plt.legend(handles=[linear_filter, perceptron])
plt.show()

##
wn_f = Wn_filtro[:, -1]
wn_p = Wn_perceptron[:, -1]

N_block = int(1e3)
blocks = int(1e3)

decisaob_f = np.zeros((N_block, 1))
decisaob_p = np.zeros((N_block, 1))
erros_dec_f = 0
erros_dec_p = 0

# uso do perceptron e do neuronio linear treinados
for b in range(0, blocks):
    # geracao dos dados do bloco
    sn_N = np.sign(np.random.randn(N_block, 1))
    vn_N = math.sqrt(sigma2_v)*(np.random.randn(N_block, 1))
    sch_N = lfilter(h_canal, 1, sn_N, axis=0)
    xn_N = sch_N + vn_N

    # organiza os dados na matriz Xn [N]:[M]
    Xn = cria_Xn_equalizacao.equaliza(xn_N, N_block, M)
    Xn_p = np.vstack((Xn, np.ones((1, N_block))))

    for n in range(0, N_block):
        dn = sn_N[n, 0]
        xn = Xn[:, n]
        xn_p = Xn_p[:, n]

        # filtro linear
        decisaob_f[n, 0] = np.sign(np.matrix(wn_f) * np.matrix(xn).T)

        # perceptron
        decisaob_p[n, 0] = np.sign(
            np.matrix(wn_p) * np.matrix(xn_p).T)

    erros_dec_f = erros_dec_f + np.sum(decisaob_f != sn_N)
    erros_dec_p = erros_dec_p + np.sum(decisaob_p != sn_N)
    if not b % 100:
        print('Bloco ', b)


Ber_f = erros_dec_f/(N_block*blocks)
Ber_p = erros_dec_p/(N_block*blocks)

print('\nBER filtro     = %.4e\n', Ber_f)
print('BER perceptron = %.4e\n', Ber_p)
print('Num. de simbolos transmitidos = %d\n', N_block*blocks)
