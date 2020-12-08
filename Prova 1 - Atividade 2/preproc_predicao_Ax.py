# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:59:44 2020

@author: Rafael
"""

import numpy as np
import scipy.io
from math import floor

def preproc(f='dados_Ax.mat'):
    '''
    Este programa carrega os dados de sensor de aceleracao do eixo x de um
    sensor triaxial montado no peito de um voluntario da base de dados
    "Activity Recognition from single chest-mounted accelerometer"
    , ja preprocessado com filtros e divide-o em pacotes para
    treinamento, validacao e teste
    '''
    mat = scipy.io.loadmat(f)
      
    xtotal = mat['sinal_Ax']
    xtotal = xtotal.flatten()   # garante que xtotal eh um vetor (com rank 1)
    N = len(xtotal)             # num. de elementos de xtotal
    
    Ntrain = floor(N*0.6)   # ~= 60% das amostras
    Nval = floor(N*0.2)     # ~= 20% das amostras
    #Ntest = N-Ntrain-Nval   # ~= 20% das amostras
    
    Mval = max(xtotal)
    mval = min(xtotal)
    xtotal = -1.+2*(xtotal-mval)/(Mval-mval)    # broadcast do mapeamento [-1,1]
                                                # note que ja retira a media tb
    
    #---- Conjunto de dados para treinamento: Ntrain amostras temporais
    x = xtotal[1:Ntrain]
    #---- Conjunto de dados para validacao: Nval amostras temporais
    xval = xtotal[Ntrain:Ntrain+Nval]
    #---- Conjunto de teste ----------------
    xtest = xtotal[Ntrain+Nval:]
    
    return x, xval, xtest