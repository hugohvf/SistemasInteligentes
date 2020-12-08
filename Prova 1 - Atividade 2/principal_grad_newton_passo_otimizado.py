# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:35:18 2020

Implementacao do metodo do gradiente e de Newton modificado nas
funcoes objetivos bidimensionais (quadratica e nao-quadratica)
@author: Rafael
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#import matplotlib.cm as cm

# modulos usados na aprendizagem da disciplina
import functions_6694 as myfun  # carrega funcoes para otimizacao utilizadas na disciplina
import optimize_6694 as myopt  # carrega funcoes e metodos de otimizacao desenvolvidos

# %% Escolha da funcao analisada
flag_f = 'fnq'
if flag_f == 'fq':
    f = myfun.fx_P3
    calc_d = myfun.calc_d_P3
    fc = myfun.fx_P3_contour
elif flag_f == 'fnq':
    f = myfun.fx_P4
    calc_d = myfun.calc_d_P4
    fc = myfun.fx_P4_contour
else:
    raise Exception('Escolha flag_f=\'fq\' para funcao quadratica\
                    e flag_f=\'fnq\' para funcao nao-quadratica!')

# %% Definicao de parametros e tipo de busca
Tolx = 1e-6     # tolerancia para a diferenca dos parametros de busca
Tolgrad = 1e-3  # tolerancia para a norma do gradiente
xmin = np.array([1. , -1.])   # xmin = [x1min;x2min]
xmax = np.array([5. , 3.])   # xmax = [x1max;x2max]
flagmetodo = 1  # 0 = metodo do gradiente 1 = metodo de Newton

Niter_max = 100
x = np.zeros((Niter_max,2))   # alocacao de memoria
k=0
x[k,0:2]= np.array([5.,2.])    # ponto inicial para teste
d,gradiente = calc_d(x[k,0:2],flagmetodo)
dfact =  myopt.direcao_factivel(x[k,:],d,xmin,xmax,Tolx)
#dfact = d.copy()

grad_anterior = np.zeros(2)
deltagrad = gradiente-grad_anterior  # para uso auxiliar no criterio de parada

while LA.norm(gradiente)>=Tolgrad and k<Niter_max-1: #and La.norm(deltagrad)>=0.001*Tolgrad #and max(abs(x(k,:)-x(k-1,:)))>=Tolx:
    # calculo do passo alpha
    alphamax = myopt.calc_alpha_max(x[k,:],dfact,xmin,xmax)
    intervalo = [0.,alphamax]
    alphaotimo = myopt.razao_aurea(f,x[k,:],dfact,intervalo,Tolx)
    # atualizacao da posicao
    # alphaotimo = 1.0
    x[k+1,:] = x[k,:] + alphaotimo*dfact # atualiza posicao
    k += 1  # atualiza contagem
    grad_anterior = gradiente.copy()
    (d,gradiente) = calc_d(x[k,:],flagmetodo)
    dfact = myopt.direcao_factivel(x[k,:],d,xmin,xmax,Tolx) # d
    #dfact = d.copy()
    deltagrad = gradiente-grad_anterior


x = x[0:k+1,:]

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
F = fc(X1.T,X2.T)

# plota curva de nivel
plt.figure( )
CS = plt.contour(X1, X2, F, 30, extent=[x1min, x1max, x2min, x2max])
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Curvas de nível da função')
plt.xlabel('$x_{2}$')
plt.ylabel('$x_{1}$')
# trajetória da solução
plt.plot(x[:,1],x[:,0],'b-o')

#delimitacao da regiao de busca permitida
ptos=10
linhax1max = np.array([xmax[0]*np.ones(ptos),
                        np.linspace(xmin[1],xmax[1],ptos)])
linhax1min = np.array([xmin[0]*np.ones(ptos),
                        np.linspace(xmin[1],xmax[1],ptos)])
linhax2max = np.array([xmax[1]*np.ones(ptos),
                        np.linspace(xmin[0],xmax[0],ptos)])
linhax2min = np.array([xmin[1]*np.ones(ptos),
                       np.linspace(xmin[0],xmax[0],ptos)])
    
plt.plot(linhax1max[1],linhax1max[0],'k',linewidth=2)
plt.plot(linhax1min[1],linhax1min[0],'k',linewidth=2)
plt.plot(linhax2max[0],linhax2max[1],'k',linewidth=2)
plt.plot(linhax2min[0],linhax2min[1],'k',linewidth=2)