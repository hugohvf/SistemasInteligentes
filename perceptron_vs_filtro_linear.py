# -*- coding: utf-8 -*-
"""
Perceptron vs filtro linear utilizando LMS
Problema de equalizacao de canal
Created on Mon Nov  2 16:14:19 2020

@author: 
"""

import numpy as np
import math
from scipy.signal import lfilter
import cria_Xn_equalizacao

M = int(3);  # dimensao de dados de entrada (portanto do filtro tambem)
N = int(1e3);   # num. de sinais transmitidos (num. de padroes) no aprendizado
mu = 0.01;

sigma2_v = 0.05;   # variancia do ruido
h_canal = [1. , .4];    # h = 1 + 0.4*z-1

# geracao dos dados
sn_N = np.sign(np.random.randn(N,1));
vn_N = math.sqrt(sigma2_v)*(np.random.randn(N,1));
sch_N = lfilter(h_canal,1,sn_N, axis=0);
xn_N = sch_N + vn_N;

# organiza os dados na matriz Xn [M]:[N]
Xn = cria_Xn_equalizacao.equaliza(xn_N,N,M);
Xn_p = np.vstack((Xn , np.ones((1,N))));  # ones eh o bias (que nao tem ruido)
