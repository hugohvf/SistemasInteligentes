# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 04:52:08 2019

@author: Rafael
"""

import math
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'Roboto',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


K = 0.15    # constante de ajuste da funcao
alpha = (-1+math.sqrt(5))/2  # numero de ouro
Tol = 1e-6
xmin = 0.3
xmax = 5

# funcao a ser avaliada
def f(x):
    return (math.sqrt(x)+math.exp(-K*x))*(x**2-6*x+10)

# Alocacao de memoria
N = 100
a = np.zeros((1,N))
b = np.zeros((1,N))
vlambda = np.zeros((1,N))
vmi = np.zeros((1,N))
f_lambda = np.zeros((1,N))
f_mi = np.zeros((1,N))

# primeira iteracao
kk=0
a[0,kk] = xmin
b[0,kk] = xmax

vlambda[0,kk] = a[0,kk] + (1-alpha)*(b[0,kk]-a[0,kk])
vmi[0,kk] = a[0,kk] + alpha*(b[0,kk]-a[0,kk])

cont_f = 0

f_lambda[0,kk] = f(vlambda[0,kk])   # avalia f em lambda
cont_f = cont_f + 1
f_mi[0,kk] = f(vmi[0,kk])           # avalia f em mi
cont_f = cont_f + 1

while (b[0,kk]-a[0,kk]) >= Tol:
        if f_lambda[0,kk] > f_mi[0,kk]:
            a[0,kk+1] = vlambda[0,kk]
            b[0,kk+1] = b[0,kk]
            vlambda[0,kk+1] = vmi[0,kk]
            vmi[0,kk+1] = a[0,kk+1] + alpha*(b[0,kk+1]-a[0,kk+1])
            f_lambda[0,kk+1] = f_mi[0,kk]
            f_mi[0,kk+1] = f(vmi[0,kk+1])
            cont_f = cont_f + 1
            kk = kk + 1
        else:
            a[0,kk+1] = a[0,kk]
            b[0,kk+1] = vmi[0,kk]
            vmi[0,kk+1] = vlambda[0,kk]
            vlambda[0,kk+1] = a[0,kk+1] + (1-alpha)*(b[0,kk+1]-a[0,kk+1])
            f_lambda[0,kk+1] = f(vlambda[0,kk+1])
            cont_f = cont_f + 1
            f_mi[0,kk+1] = f_lambda[0,kk]
            kk = kk + 1

# corte da alocação não utilizada
a = a[0,0:kk]
b = b[0,0:kk]

#%% plota resultados
eixo_k = np.arange(1,kk+1)
estimativa_final = (a[kk-1] + b[kk-1])/2
intervalo_final = b[kk-1]-a[kk-1]

print("Valor final: ", estimativa_final)
print("Intervalo final: ", intervalo_final)
print("Num. de avaliações: ", cont_f)

ak_evo, = plt.plot(eixo_k,a,'b-o', label="$a^k$")
bk_evo, = plt.plot(eixo_k,b,'r-*', label="$b^k$")
plt.xlabel("iteração k", fontdict=font)
plt.ylabel("$a^k$, $b^k$", fontdict=font)
plt.title("Evolução do intervalo", fontdict=font)
plt.legend(handles=[ak_evo,bk_evo])
plt.show()

#%% plota função na qual estamos buscando o mínimo
eixo_x = np.arange(xmin,xmax,1e-3)
def f_vec(x):
    return (np.sqrt(x)+np.exp(-K*x))*(np.power(x,2)-6*x+10)

plt.figure()
plt.plot(eixo_x,f_vec(eixo_x))
plt.xlabel("Eixo x: variável de busca", fontdict=font)
plt.ylabel("f(x)", fontdict=font)
plt.title("Função objetivo", fontdict=font)
plt.show()