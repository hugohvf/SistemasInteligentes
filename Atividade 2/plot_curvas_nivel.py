# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:59:06 2019

@author: rkrummen
"""

import math
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm       # colormap
from mpl_toolkits.mplot3d import Axes3D # <--- for 3d plotting

# Este script plota as curvas de nivel funcao custo (objetivo)
# Função não-linear quadrática
# f(x) = (x1-3)^2+(x2-1)^2+(x1-3)*(x2-1)
def fx_P3(x1,x2):
    return (x1-3)**2+(x2-1)**2+(x1-3)*(x2-1)

# Função não-linear não quadrática
# f(x)= exp(-k*x2)*(x1-1)^2+exp(-k*x1)*(x2-2)^2
def fx_P4(x1,x2):
    k=0.5
    return math.exp(-k*x2)*(x1-1)**2+math.exp(-k*x1)*(x2-2)**2

flag_f = 'fnq'
if flag_f=='fq':
    f = fx_P3
elif flag_f=='fnq':
    f = fx_P4
else:
    raise Exception('Escolha flag_f=\'fq\' para função quadrática\
                    e flag_f=\'fnq\' para função não-quadrática!')

x1min,x1max = -1, 5

x2min,x2max = -1, 5

ptos = 100

faixa_x1 = np.linspace(x1min, x1max, ptos)
faixa_x2 = np.linspace(x2min, x2max, ptos)

F = np.zeros((ptos,ptos)) # alocacao de memoria para os pontos de fx

# varredura na grade de pontos para avaliar fx
for p in range(ptos):
    for r in range(ptos):
        F[p,r]= f(faixa_x1[p], faixa_x2[r])        


# meshgrid para plotar contour e surf
X1, X2 = np.meshgrid(faixa_x1, faixa_x2)
fig, f1 = plt.subplots()
# Contour
CS = f1.contour(X1, X2, F, 30)
f1.clabel(CS, inline=1, fontsize=10)
f1.set_title('Curvas de nível da função quadrática')
plt.xlabel('$x_{2}$')
plt.ylabel('$x_{1}$')
if flag_f=='fq':
    plt.plot(np.array([1.]),np.array([3.]),'*',markersize=12)
else:
    plt.plot(np.array([2.]),np.array([1.]),'*',markersize=12)


# Surface
fig = plt.figure()
f1 = fig.gca(projection='3d')
# Plot the surface
surf = f1.plot_surface(X1,X2,F,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

f1.clabel(CS, inline=1, fontsize=10)
f1.set_title('Superficie da função quadrática')
plt.xlabel('$x_{2}$')
plt.ylabel('$x_{1}$')

