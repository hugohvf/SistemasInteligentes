# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:43:09 2020

Uso da scipy.optimize para otimizacao caixa-preta (usando
as funcoes da biblioteca)
Para detalhes, veja:
http://scipy-lectures.org/advanced/mathematical_optimization/index.html

@author: Rafael Krummenauer
"""

import time # para medicao de tempo com time.time()

import numpy as np
from scipy import optimize
from math import exp
from numpy import linalg as LA

# Definicao das funcoess
# ======= funcoes com n=2 variaveis de busca ==============
# Funcao nao-linear quadratica
# f(x) = (x1-3)^2+(x2-1)^2+(x1-3)*(x2-1)
def fx_P3(x):
    return (x[0]-3)**2+(x[1]-1)**2+(x[0]-3)*(x[1]-1)

# Funcao nao-linear nao-quadratica
# f(x)= exp(-k*x2)*(x1-1)^2+exp(-k*x1)*(x2-2)^2
def fx_P4(x):
    x1 = x[0]
    x2 = x[1]
    k=0.5
    return np.exp(-k*x2)*(x1-1)**2+np.exp(-k*x1)*(x2-2)**2

# funcao que descreve o gradiente (Jacobiano) da funcao quadratica
def grad_P3(x):
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*(x1-3)+x2-1 , 2*(x2-1)+x1-3]) # rank 1
    
    return gradiente

# funcao que descreve a Hessiana da funcao quadratica
def Hess_P3(x):
    Hessiana = np.array([[2., 1.],[1., 2.]])
    return Hessiana

# calcula o gradiente (Jacobiano) da funcao nao-quadratica
def grad_P4(x):
    k = 0.5
    x1 = x[0]
    x2 = x[1]
    ## calculados analiticamente
    gradiente = np.array([2*exp(-k*x2)*(x1-1)-k*exp(-k*x1)*(x2-2)**2, 
                          -k*exp(-k*x2)*(x1-1)**2+ 2*exp(-k*x1)*(x2-2)])
    
    return gradiente

# funcao que descreve a Hessiana da funcao nao-quadratica
def Hess_P4(x):
    k = 0.5
    x1 = x[0]
    x2 = x[1]
    Hessiana = np.array([[2*exp(-k*x2)+k**2*exp(-k*x1)*(x2-2)**2, 
                          -2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2)], 
                            [-2*k*exp(-k*x2)*(x1-1)-2*k*exp(-k*x1)*(x2-2), 
                             k**2*exp(-k*x2)*(x1-1)**2+2*exp(-k*x1)]])
    return Hessiana

# Definicao da funcao analisada
flag_f = 'fnq'
if flag_f == 'fq':
    f = fx_P3
    gradient = grad_P3
    Hessian = Hess_P3
elif flag_f == 'fnq':
    f = fx_P4
    gradient = grad_P4
    Hessian = Hess_P4
else:
    raise Exception('Escolha flag_f=\'fq\' para funcao quadratica\
                    e flag_f=\'fnq\' para funcao nao-quadratica!')

# %% Otimizacao

#opt_methods = ["CG","Newton-CG","BFGS","L-BFGS-B"] 

# ponto de inicializacao
x_ini = np.array([5., 2.])
Ntimes = 1000
print('Tempo referente a N = %d repeticoes de cada processo de busca' % Ntimes)
# Gradiente Conjugado
t0 = time.time()
for i in range(Ntimes):
    x_cg = optimize.minimize(f, x_ini, method="CG")
print('           CG: time %.3fs, x=[%.4f,%.4f], nfev %d' % (time.time() - t0,
      x_cg.x[0],x_cg.x[1], x_cg.nfev))

# Newton-CG sem Hessiana
t0 = time.time()
for i in range(Ntimes):
    x_ncg_sem_H = optimize.minimize(f, x_ini, method="Newton-CG", 
                          jac=gradient)
print('    Newton-CG: time %.3fs, x=[%.4f,%.4f], nfev %d' % (time.time() - t0,
      x_ncg_sem_H.x[0],x_ncg_sem_H.x[1], x_ncg_sem_H.nfev))

# Newton-CG com Hessiana
t0 = time.time()
for i in range(Ntimes):
    x_ncg_com_H = optimize.minimize(f, x_ini, method="Newton-CG", 
                          jac=gradient, hess=Hessian)
print('Newton-CG w H: time %.3fs, x=[%.4f,%.4f], nfev %d' % (time.time() - t0,
      x_ncg_com_H.x[0],x_ncg_com_H.x[1], x_ncg_com_H.nfev))

t0 = time.time()
for i in range(Ntimes):
    x_bfgs_sem_jac = optimize.minimize(f, x_ini, method="BFGS")
print('  BFGS sem f\': time %.3fs, x=[%.4f,%.4f], nfev %d' % (time.time() - t0,
      x_bfgs_sem_jac.x[0],x_bfgs_sem_jac.x[1], x_bfgs_sem_jac.nfev))

t0 = time.time()
for i in range(Ntimes):
    x_bfgs_com_jac = optimize.minimize(f, x_ini, method="BFGS", jac=gradient)
print('  BFGS com f\': time %.3fs, x=[%.4f,%.4f], nfev %d' % (time.time() - t0,
      x_bfgs_com_jac.x[0],x_bfgs_com_jac.x[1], x_bfgs_com_jac.nfev))