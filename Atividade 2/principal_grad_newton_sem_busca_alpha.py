# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:57:38 2019

@author: rkrummen
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm

from math import exp
from numpy import linalg as LA

#import calc_d

# ======= funcoes ===============
# Função não-linear quadrática
# f(x) = (x1-3)^2+(x2-1)^2+(x1-3)*(x2-1)
def fx_P3(x1,x2):
    return (x1-3)**2+(x2-1)**2+(x1-3)*(x2-1)

# Função não-linear não quadrática
# f(x)= exp(-k*x2)*(x1-1)^2+exp(-k*x1)*(x2-2)^2
def fx_P4(x1,x2):
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

# Definicao da funcao analisada
flag_f = 'fq'
if flag_f == 'fq':
    f = fx_P3
    calc_d = calc_d_P3
elif flag_f == 'fnq':
    f = fx_P4
    calc_d = calc_d_P4
else:
    raise Exception('Escolha flag_f=\'fq\' para função quadrática\
                    e flag_f=\'fnq\' para função não-quadrática!')

Tolx = 1e-6     # tolerancia para a diferenca dos parametros de busca
Tolgrad = 1e-3  # tolerancia para a norma do gradiente
xmin = np.array([-1.0 , -1.0])   # xmin = [x1min;x2min]
xmax = np.array([5.0 , 5.0])   # xmax = [x1max;x2max]
#flagpasso = 0   # 1 = busca unidimensional do passo, 0 = sem busca (alpha=1), 2 = backtracking
flagmetodo = 0  # 0 = metodo do gradiente 1 = metodo de Newton

Niter_max = 200
x = np.zeros((Niter_max,2))   # alocacao de memoria
k=0
x[k,0:2]= np.array([5,2])    # ponto inicial para teste
d,gradiente = calc_d(x[k,0:2],flagmetodo)
#dfact =  direcao_factivel(x(k,:),d,xmin,xmax,Tolx)
dfact = d.copy()

grad_anterior = np.zeros(2)
deltagrad = gradiente-grad_anterior  # para uso auxiliar no criterio de parada

alphaotimo = 0.3
while LA.norm(gradiente)>=Tolgrad and k<Niter_max-1: #and La.norm(deltagrad)>=0.001*Tolgrad #and max(abs(x(k,:)-x(k-1,:)))>=Tolx:
    x[k+1,:] = x[k,:] + alphaotimo*dfact # atualiza posicao
    k += 1  # atualiza contagem
    grad_anterior = gradiente.copy()
    (d,gradiente) = calc_d(x[k,:],flagmetodo)
    #dfact = direcao_factivel(x(k,:),d,xmin,xmax,Tolx) # d
    dfact = d.copy()
    deltagrad = gradiente-grad_anterior


print('Número de passos para convergência: {}'.format(k))
print('Solução final: [{} , {}]'.format(x[k,0],x[k,1]))

# plot contorno e evolução da solução da técnica
x1min,x1max = -1, 5
x2min,x2max = -1, 5

ptos = 100

faixa_x1 = np.linspace(x1min, x1max, ptos)
faixa_x2 = np.linspace(x2min, x2max, ptos)
# meshgrid para varrer toda a grade de x1 e x2 nas
# matrizes de malha X1 e X2
X1, X2 = np.meshgrid(faixa_x1, faixa_x2)

# Em apenas uma linha o cômputo é vetorizado/broadcast
F = f(X1.T,X2.T)

# plota curva de nivel
plt.figure( )
CS = plt.contour(X1, X2, F, 30, extent=[x1min, x1max, x2min, x2max])
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Curvas de nível da função')
plt.xlabel('$x_{2}$')
plt.ylabel('$x_{1}$')
# trajetória da solução
plt.plot(x[:,1],x[:,0],'b-o')