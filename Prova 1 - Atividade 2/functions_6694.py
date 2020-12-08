# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:34:32 2020

Modulo de funcoes para aprendizado em otimizacao da
 disciplina 6694 - Sistemas inteligentes.
Contem os metodos:
    fx_P3(x): calcula valor da funcao quadratica
    fx_P4(x): calcula valor da funcao nao-quadratica
    fx_P3_contour(x1,x2): broadcast de fx_P3 para meshgrid
    fx_P4_contour(x1,x2): broadcast de fx_P4 para meshgrid
    calc_d_P3(x,flagmetodo=0): calcula direcao d e o 
        gradiente da funcao fx_P3
    calc_d_P4(x,flagmetodo=0): calcula direcao d e o 
        gradiente da funcao fx_P4

@author: Rafael
"""

from math import exp
import numpy as np
from numpy import linalg as LA

# ======= funcoes ===============
# Funcao nao-linear quadratica
# f(x) = (x1-3)^2+(x2-1)^2+(x1-3)*(x2-1)
def fx_P3(x):
    x1 = x[0]
    x2 = x[1]
    return (x1-3)**2+(x2-1)**2+(x1-3)*(x2-1)

def fx_P3_contour(x1,x2):
    return (x1-3)**2+(x2-1)**2+(x1-3)*(x2-1)

# Funcao nao-linear nao quadratica
# f(x)= exp(-k*x2)*(x1-1)^2+exp(-k*x1)*(x2-2)^2
def fx_P4(x):
    x1 = x[0]
    x2 = x[1]
    k=0.5
    return np.exp(-k*x2)*(x1-1)**2+np.exp(-k*x1)*(x2-2)**2

def fx_P4_contour(x1,x2):
    k=0.5
    return np.exp(-k*x2)*(x1-1)**2+np.exp(-k*x1)*(x2-2)**2

# calcula o gradiente da funcao quadratica
def calc_d_P3(x,flagmetodo=0):
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*(x1-3)+x2-1 , 2*(x2-1)+x1-3]) # rank 1
    Hessiana = np.array([[2., 1.],[1., 2.]])
    
    if flagmetodo == 0:
        d = -gradiente/(LA.norm(gradiente))
    else:
        d = LA.solve(-Hessiana,gradiente)    # inv(Hessiana)*gradiente
        
    return d, gradiente

# calcula o gradiente da funcao nao-quadratica
def calc_d_P4(x,flagmetodo=0):
    k = 0.5
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*exp(-k*x2)*(x1-1)-k*exp(-k*x1)*(x2-2)**2, 
                          -k*exp(-k*x2)*(x1-1)**2+ 2*exp(-k*x1)*(x2-2)])
    
    Hessiana = np.array([[2*exp(-k*x2)+k**2*exp(-k*x1)*(x2-2)**2, 
                          -2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2)], 
                            [-2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2), 
                             k**2*exp(-k*x2)*(x1-1)**2+2*exp(-k*x1)]])
    if flagmetodo == 0:
        d = -gradiente/(LA.norm(gradiente))
    else:
        d = LA.solve(-Hessiana,gradiente)    # inv(Hessiana)*gradiente
        
    return d, gradiente

