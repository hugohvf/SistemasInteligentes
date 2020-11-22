# -*- coding: utf-8 -*-
"""

Função de equalização Xn
Created on Sun Nov  1 15:33:24 2020
@author:Hugo Fusinato
"""
import numpy as np

# Calc alphamax


def equalize(xn_N, M):

    xn_N = np.matrix(xn_N)

    Xn = np.zeros((M, len(xn_N)))
    for n in range(len(xn_N)):
        if n < M - 1:
            Xn[0 : n + 1, n] = xn_N[0 : n + 1, :][::-1].reshape(n+1)
        if n >= M - 1:
            Xn[:, n] = xn_N[n-2:n+1, :][::-1].reshape(3)

    return Xn
