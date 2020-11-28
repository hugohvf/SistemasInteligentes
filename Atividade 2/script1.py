# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 00:07:52 2020

6694 - Sistemas inteligentes

Exercício computacional 1: Algebra básica,
 mapeamento entrada-saída, norma e sintaxe.

@author: Rafael
"""
import numpy as np

dim = int(1e2)      # dimensao
ndata = int(1e4)  # num. de dados/vetores

# crio os dados aleatoriamente, dimensao dim x ndata
d = np.random.randn(dim,ndata)

#A = np.eye(dim)   # exemplo de transformação linear trivial
A = 1.-2.*np.random.random((dim,dim))

# alocacao de memória para a saída
Y = np.zeros((dim,ndata))

# transformação linear
for i in range(ndata):
    Y[:,i] = np.dot(A,d[:,i])

# métrica de comparação
e2 = 0.
for i in range(ndata):
    tmp = Y[:,i]-d[:,i]
    e2 = e2 + np.dot(tmp,tmp)
    
print(e2/ndata)