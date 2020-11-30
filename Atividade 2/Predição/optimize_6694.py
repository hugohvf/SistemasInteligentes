# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:40:23 2020

Modulo de otimizacao da disciplina 6694 - Sistemas
 inteligentes.
Classe optimize_6694 com os metodos:
    fx_P3(x): calcula valor da funcao quadratica
    fx_P4(x): calcula valor da funcao nao-quadratica
    fx_P3_contour(x1,x2): broadcast de fx_P3 para meshgrid
    fx_P4_contour(x1,x2): broadcast de fx_P4 para meshgrid
    calc_d_P3(x,flagmetodo=0): calcula direcao d e o 
        gradiente da funcao fx_P3
    calc_d_P4(x,flagmetodo=0): calcula direcao d e o 
        gradiente da funcao fx_P4
    calc_alpha_max(xk,dk,xmin,xmax): calcula passo maximo
        dado os limites xmin e xmax das variaveis
    razao_aurea(f,xk,dk,intervalo,Tol): metodo de busca
        da secao aurea no intervalo definido [0,alphamax]

@author: Rafael
"""
from math import sqrt
import numpy as np

# Calculo do passo maximo (alphamax)
def calc_alpha_max(xk,dk,xmin,xmax):
    n = len(xmin)
    alpha = np.zeros(n)

    for ii in range(n):
        if dk[ii]>0.:
            alpha[ii] = (xmax[ii]-xk[ii])/dk[ii]
        elif dk[ii]<0.:
            alpha[ii] = (xmin[ii]-xk[ii])/dk[ii]
        elif dk[ii] == 0.:
            alpha[ii] = 1e9
    
    if all(np.equal(alpha,1e9*np.ones(n))):
        alpha = np.zeros(n)
    
    return min(alpha)   # alphamax

# Busca em linha pelo metodo da secao aurea
def razao_aurea(f,xk,dk,intervalo,Tol):
    raurea = (-1.+sqrt(5.))/2. # razao aurea = 0.618

    a,b,vlambda,vmi,f_lambda,f_mi = list(),list(),list(),list(),list(),list()
    kk=0
    a.append(intervalo[0]) # = 0, limite inferior do intervalo
    b.append(intervalo[1]) # = alphamax, limite superior do intervalo

    vlambda.append(a[kk] + (1.-raurea)*(b[kk]-a[kk]))
    vmi.append(a[kk] + raurea*(b[kk]-a[kk]))

    f_lambda.append(f(xk+vlambda[kk]*dk))
    f_mi.append(f(xk+vmi[kk]*dk))
    
    while (b[kk]-a[kk]) >= Tol:
        if f_lambda[kk] > f_mi[kk]:
            a.append(vlambda[kk])   # a[k+1] = vlambda[k]
            b.append(b[kk])         # b[k+1] = b[k]
            vlambda.append(vmi[kk])
            vmi.append( a[kk+1] + raurea*(b[kk+1]-a[kk+1]))
            f_lambda.append(f_mi[kk])
            f_mi.append(f(xk+vmi[kk+1]*dk))
            kk = kk + 1
        else:
            a.append(a[kk])         # a[k+1] = a[k]
            b.append(vmi[kk])       # b[k+1] = vmi[k]
            vmi.append(vlambda[kk])
            vlambda.append(a[kk+1] + (1.-raurea)*(b[kk+1]-a[kk+1]))
            f_lambda.append(f(xk+vlambda[kk+1]*dk))
            f_mi.append(f_lambda[kk])
            kk = kk + 1
        
    return (a[kk] + b[kk])/2. # alphaotimo

# regiao de factibilidade
def direcao_factivel(xk,dk,xmin,xmax,Tolx):
    n = len(xk)
    dfact = np.zeros(n)
    for ii in range(n):
        if abs(xk[ii] - xmax[ii])<Tolx and dk[ii]>0.:
            dfact[ii] = 0.  # borda superior e dk positiva, entao paro de andar nesta dimensao
        elif abs(xk[ii] - xmin[ii])<Tolx and dk[ii]<0.:
            dfact[ii] = 0.  # borda inferior e dk negativa, entao paro de andar nesta dimensao
        else:
            dfact[ii] = dk[ii]
    
    return dfact

def myBFGSstep(f,calc_d,vxi,vgi,mHi,niter,dim,Tol,xmin,xmax):

    vdi = np.dot(mHi,-vgi)   # vetor de direcao na iteracao i vdi:[dim]
    if ((niter+1 % dim) == 0):     # verifica loop para reset da matriz mHi apos dim iteracoes
        vdi = -vgi
        mHi = np.eye(dim)
    
    vdi = direcao_factivel(vxi,vdi,xmin,xmax,Tol) # vdi com direcao factivel
    
    #intervalo = [0.,1.] % intervalo de busca do alpha com alphamax=1
    alphamax = calc_alpha_max(vxi,vdi,xmin,xmax)
    intervalo = [0.,alphamax]

    alphaotimo = razao_aurea(f,vxi,vdi,intervalo,Tol) # busca unidimensional em alpha. Tol=tolerancia do intervalo de incerteza.

    vxip1 = vxi + alphaotimo*vdi  # vxip1:[dim] = vetor x(i+1)
    # Calculo de mHip1 e vgip1
    vpi = alphaotimo*vdi # vetor pi usado para calculo de Hi
    _,vgip1 = calc_d(vxip1,0) # rotina de calculo do novo gradiente
    vqi = vgip1-vgi # vqi:[dim], vetor qi usado para calculo de Hi
    piTqi = np.dot(vpi,vqi)   # escalar util para evitar calculos redundantes em Hi
    
    mHip1 = mHi + (np.outer(vpi,vpi)/piTqi)*(1.+ np.dot(vqi,np.dot(mHi,vqi))/piTqi)
    -(np.dot(mHi,np.outer(vqi,vpi)) + np.dot(np.outer(vpi,vqi),mHi))/piTqi # mHip1 = H(i+1):[dim]x[dim]
    
    return vxip1,vgip1,mHip1