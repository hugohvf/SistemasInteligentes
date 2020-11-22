# -*- coding: utf-8 -*-
"""

Perceptron vs filter linear utilizando LMS
Problema de equalizacao de canal
Created on Sun Nov  1 17:34:58 2020

@author: Hugo Fusinato
"""

import alphamax
from scipy.signal import lfilter
import math
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
%matplotlib inline

font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }

M = int(3)
N = int(1e3)
mu = 0.01

variance_noise = 0.05
h_channel = [1., .4]

# Generating filter and perceptron
vn_N = math.sqrt(variance_noise)*(np.random.randn(N, 1))
sn_N = np.sign(np.random.randn(N, 1))
sch_N = lfilter(h_channel, 1, sn_N, axis=0)
xn_N = sch_N + vn_N
antes = xn_N
Xn = alphamax.equalize(xn_N, M)
depois = Xn
Xn_p = np.vstack((Xn, np.ones((1, N))))

error_powred_filter = np.zeros((N, 1))
error_powred_perceptron = np.zeros((N, 1))
Wn_filter = np.zeros((M, N+1))
Wn_perceptron = np.zeros((M+1, N+1))
Yn_filter = np.zeros((N, 1))
Yn_perceptron = np.zeros((N, 1))

for n in range(N):
    dn = sn_N[n, 0]
    xn = Xn[:, n]
    xn_p = Xn_p[:, n]

    wn = Wn_filter[:, n]
    yn = np.matrix(wn) * np.matrix(xn).T
    erro = dn - yn
    Wn_filter[:, n+1] = wn + mu * erro * xn
    error_powred_filter[n, 0] = np.power(erro, 2)
    Yn_filter[n, 0] = yn

    wn = Wn_perceptron[:, n]
    yn = np.sign(np.matrix(wn) * np.matrix(xn_p).T)
    erro = dn - yn
    Wn_perceptron[:, n+1] = wn + mu*erro*xn_p
    error_powred_perceptron[n, 0] = np.power(erro, 2)
    Yn_perceptron[n, 0] = yn

vetor_n = np.array(np.arange(1, N + 1))

linear_neuronion_plot, = plt.plot(
    vetor_n, error_powred_filter, 'c', label="neuronio linear")
perceptron_plot, = plt.plot(
    vetor_n, error_powred_perceptron, 'm', label="perceptron")
plt.title("Curvas de erro quadrático", fontdict=font)
plt.legend(handles=[linear_neuronion_plot, perceptron_plot])
plt.show()

plt.plot(vetor_n, sch_N, 'o')
plt.xlabel("Iterations (n)", fontdict=font)
plt.ylabel("Channel output", fontdict=font)
plt.title("Channel output (without noise)", fontdict=font)
plt.show()

plt.plot(vetor_n, xn_N, '.')
plt.xlabel("Iterations (n)", fontdict=font)
plt.ylabel("Channel output", fontdict=font)
plt.title("Channel output (with noise)", fontdict=font)
plt.show()

linear_filter, = plt.plot(vetor_n, Yn_filter, '.', label="Linear filter")
perceptron_plot, = plt.plot(vetor_n, Yn_perceptron, '.', label="Perceptron")
plt.xlabel("Iterations (n)", fontdict=font)
plt.ylabel("Channel output", fontdict=font)
plt.title("Equalizer output", fontdict=font)
plt.legend(handles=[linear_filter, perceptron_plot])
plt.show()

fig, ax = plt.subplots()
for i in range(M):
    ax.plot(vetor_n, Wn_filter[i, 0:-1], label=r"$F_%d(x)$" % i)
plt.xlabel("Iterations (n)", fontdict=font)
plt.ylabel("Weights", fontdict=font)
plt.title("Evolution from filters weight", fontdict=font)
ax.legend()
plt.show()

fig, ax = plt.subplots()
for i in range(M):
    ax.plot(vetor_n, Wn_perceptron[i, 0:-1], label=r"$P_%d(x)$" % i)
plt.xlabel("Iterations (n)", fontdict=font)
plt.ylabel("Weights", fontdict=font)
plt.title("Evolution from filters weight", fontdict=font)
ax.legend()
plt.show()

# Testing filter and perceptron
wn_f = Wn_filter[:, -1]
wn_p = Wn_perceptron[:, -1]

N_block = int(1e3)
blocks = int(1e3)

decision_filter = np.zeros((N_block, 1))
decision_perceptron = np.zeros((N_block, 1))
error_filter = 0
error_perceptron = 0

for b in range(blocks):
    sn_N = np.sign(np.random.randn(N_block, 1))
    vn_N = math.sqrt(variance_noise)*(np.random.randn(N_block, 1))
    sch_N = lfilter(h_channel, 1, sn_N, axis=0)
    xn_N = sch_N + vn_N

    Xn = alphamax.equalize(xn_N, M)
    Xn_p = np.vstack((Xn, np.ones((1, N_block))))

    for n in range(N_block):
        dn = sn_N[n, 0]
        xn = Xn[:, n]
        xn_p = Xn_p[:, n]

        decision_filter[n, 0] = np.sign(np.matrix(wn_f) * np.matrix(xn).T)

        decision_perceptron[n, 0] = np.sign(
            np.matrix(wn_p) * np.matrix(xn_p).T)

    error_filter = error_filter + np.sum(decision_filter != sn_N)
    error_perceptron = error_perceptron + np.sum(decision_perceptron != sn_N)
    if not b % 100:
        print('Block ', b)

Ber_f = error_filter/(N_block*blocks)
Ber_p = error_perceptron/(N_block*blocks)

print('\nErros filter: ', error_filter,
      '\nErros perceptron: ', error_perceptron)
print('BER filter: ', Ber_f)
print('BER perceptron: ', Ber_p)
print('Number of iterations: ', N_block*blocks)
