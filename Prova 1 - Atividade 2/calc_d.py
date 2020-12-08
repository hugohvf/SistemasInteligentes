# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:40:47 2019

@author: rkrum
"""
import numpy as np
from math import exp
from numpy import linalg as LA

# calcula o gradiente da funcao quadratica
def calc_d_P3(x,flagmetodo=0):
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*(x1-3)+x2-1 , 2*(x2-1)+x1-3]) # rank 1
    Hessiana = np.array([[2, 1],[1, 2]])
    
    if flagmetodo == 0:
        d = -gradiente/(LA.norm(gradiente))
    else:
        d = LA.solve(-Hessiana,gradiente)    # inv(Hessiana)*gradiente
        
    return d, gradiente

def calc_d_P4(x,flagmetodo=0):
    k = 0.5
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*exp(-k*x2)*(x1-1)-k*exp(-k*x1)*(x2-2)^2, \
                          -k*exp(-k*x2)*(x1-1)^2+ 2*exp(-k*x1)*(x2-2)])
    
    Hessiana = np.array([[2*exp(-k*x2)+k^2*exp(-k*x1)*(x2-2)^2, \
                          -2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2)], \
                            [-2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2), \
                             k^2*exp(-k*x2)*(x1-1)^2+2*exp(-k*x1)]])
    if flagmetodo == 0:
        d = -gradiente/(LA.norm(gradiente))
    else:
        d = LA.solve(-Hessiana,gradiente)    # inv(Hessiana)*gradiente
        
    return d, gradiente

if flag_f == 'fq':
    calc_d = calc_d_P3
elif flag_f == 'fnq':
    calc_d = calc_d_P4
else:
    raise Exception('Escolha flag_f=\'fq\' para função quadrática\
                    e flag_f=\'fnq\' para função não-quadrática!')