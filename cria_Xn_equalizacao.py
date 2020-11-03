# -*- coding: utf-8 -*-
"""
Função de equalização
Created on Mon Nov  2 14:51:12 2020

@author:
"""

# cria matriz de sinais recebidos dada a ordem M
# do filtro (canal) e o num. de amostras transmitidas N
# N = num. de sinais transmitidos no aprendizado
# M = ordem do filtro (canal)

import numpy as np

# Calculo do passo maximo (alphamax)
def equaliza(xn_N,N,M):

    n_N = np.matrix(xn_N);
    
    Xn = np.zeros((M,N));
    for n in range(0,N):
        # transiente de preenchimento dos taps (instantes) do filtro
        if n < M -1:
            Xn[0:n+1,n] = xn_N[0:n+1,:][::-1].reshape(n+1)
            
            #Xn[1:n,n] = xn_N[n:-1:1,1]; # passado para baixo [x(n);x(n-1);...;x(n-(M-1))]        
        # regime de funcionamento dos taps do filtro
        if n >= M - 1:
            Xn[:,n] = xn_N[n-2:n+1,:][::-1].reshape(3)

    return Xn