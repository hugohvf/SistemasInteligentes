# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:53:20 2020

Módulo que implementa uma MLP de uma camada intermediária

@author: Rafael
"""

import numpy as np

#%%
def inicia_pesos(Nl,m):
    '''Esta funcao gera aleatoriamente os pesos da rede neural, usando
        uma distribuicao uniforme com intervalo [-0.1,0.1].
        (w1,w2) = inicia_pesos(Nl,m)
        m = num. de entradas
        Nl = eh um vetor 1x2 contendo o num. de neuronios da camada intermediaria e da
        camada de saida, Nl = [Nl[0],Nl[1]].
        w1, w2: matrizes dos pesos (uma para cada camada)
        w1=[Nl[0]]por[m+1]   w2=[Nl[1]]por[Nl[0]+1]'''

    print('Gerando pesos iniciais da rede!')
    w1 = 0.1*(-1. + 2.*np.random.rand(Nl[0],m+1))
    w2 = 0.1*(-1. + 2.*np.random.rand(Nl[1],Nl[0]+1))
    return w1,w2

#%%
def embaralha(X,S):
    '''Esta funcao embaralha aleatoriamente a matrix de dados X
    X: [m+1]por[N]'''
    N = X.shape[1]
    rindices = np.random.permutation(N)
    Xemb = X[:,rindices]
    if S.size > S.shape[0]:
        Semb = S[:,rindices]
    else:
        Semb = S[rindices]
    
    return Xemb,Semb

#%%
def forwardprop_and_backprop_batch(X,S,w1,w2,Nl,m,N):
    '''
    % (Ew,dEw) = forwardprop_and_backprop(X,S,w1,w2,Nl,m,N)
    % Output: squared error Ew and gradient vector dEw
    % Presentation of input-output patterns: batch mode
    % All neurons have bias
    %
    % m: dimensao dos dados de entrada
    % N: numero de amostras para treinamento
    % Nl: [1]por[L], com L=2, onde cada coluna contem o numero de neuronios das camadas
    % w1: [Nl[0]]por[m+1], matriz de pesos da camada 1
    % w2: [Nl[1]]por[Nl[0]+1], matriz de pesos da camada 2 (saida)
    % X e S: dados de treinamento (X) e saidas (S) desejadas para cada padrao
    '''
    
    #---------- Em LOTE ou BATELADA ----------------------
    #%% Propaga os dados
    x1 = np.vstack([X , np.ones(N)])   # ones representa os bias. x1:[m+1]por[N] entrada camada 1
    y1 = np.tanh(w1 @ x1)       # saida das funcoes de ativacao para todos os neuronios e para todo o lote. y1:[Nl(1)]por[N]
    x2 = np.vstack([y1 , np.ones(N)])  # ones representa os bias. x2:[Nl[0]+1]por[N]
    # y2 = np.tanh(w2 @ x2)     # saida COM funcao de ativacao
    y2 = w2 @ x2           # saida SEM funcao de ativacao. y2:[Nl[1]]por[N]
    erro = y2-S            # erro na camada de saida. erro:[Nl[1]]por[N]
    
    #%% Backpropagation dos erros
    # Gradiente local da camada 2 (saida) dw2 = delta2*x2'
    # delta2 = 2.*erro*(1.0-y2*y2); # onde "(1.0-y2.*y2)" eh a derivada em
                                     # relacao a y2 da funcao de ativacao tanh(y2),
                                     # (matriz diagonal Fponto aplicada na forma de produto de hadamard).
    delta2 = 2.*erro       # nao ha funcao de ativacao na saida, portanto a derivada de y2 (f_ponto) eh unitaria.
    dw2 = (1/N)*(delta2 @ x2.T)    # gradiente local camada de saida (sensibilidade local x entrada na camada). dw2:[Nl[1])]por[Nl[0]+1]
    
    # Gradiente local da camada 1 (intermediaria) dw1 = delta1*x1'
    delta1 = (w2[:,0:Nl[0]].T @ delta2)*(1.0-y1*y1) # sensibilidade da primeira camada.
    # IMPORTANTE: w2(0:Nl[0],:) pois nao usamos os bias na retropropagacao do
    # erro. delta1:[Nl[0]]por[N]
    dw1 = (1/N)*(delta1 @ x1.T) # gradiente local da primeira camada. dw1:[Nl[0]]por[m+1]
    
    #%% Calcula erro Ew = J_N e dEw = delta J_N (gradiente)
    verro = np.reshape(erro,N*Nl[1])     # vetor de erro, verro:[N*Nl[1]]
    Ew = (0.5/N)*(verro @ verro)            # Ew:[1]x[1], escalar
    dEw = np.block([np.reshape(dw1,Nl[0]*(m+1)), np.reshape(dw2,Nl[1]*(Nl[0]+1))])   # dEw:[Nl(1)*(m+1)+Nl(2)*(Nl(1)+1)]por[1],
    # dEw tem a mesma dimensao que o num. de pesos da rede neural
    return Ew,dEw

#%%
def evalJw(vw,X,S,Nl,m,npesos,N):
    '''% Esta funcao avalia a funcao custo J(w)=0.5*verro'*verro, onde verro=y2-S
    dados w1 e w2.'''
    
    w1 = np.reshape(vw[0:Nl[0]*(m+1)],(Nl[0],m+1))
    w2 = np.reshape(vw[npesos-Nl[1]*(Nl[0])-1:npesos],(Nl[1],Nl[0]+1))
    
    # Propaga os dados
    x1 = np.vstack([X , np.ones(N)])   # ones representa os bias. x1:[m+1]por[N] entrada camada 1
    y1 = np.tanh(w1 @ x1)       # saida das funcoes de ativacao para todos os neuronios e para todo o lote. y1:[Nl(1)]por[N]
    x2 = np.vstack([y1 , np.ones(N)])  # ones representa os bias. x2:[Nl[0]+1]por[N]
    # y2 = np.tanh(w2 @ x2)     # saida COM funcao de ativacao
    y2 = w2 @ x2           # saida SEM funcao de ativacao. y2:[Nl[1]]por[N]
    erro = y2-S            # erro na camada de saida. erro:[Nl[1]]por[N]
    
    verro = np.reshape(erro,N*Nl[1])     # vetor de erro, verro:[N*Nl[1]]
    Jw = (0.5/N)*(verro @ verro)            # Jw:[1]x[1] = Ew em processa()
    return Jw

#%%
def processa(X,S,w1,w2,Nl,N):
    '''% Esta funcao processa os erros da MLP na fase de testes e devolve o sinal
    de saida estimado Y e o erro quadratico em relacao ao sinal correto S.'''
    
    # Propaga os dados
    x1 = np.vstack([X , np.ones(N)])   # ones representa os bias. x1:[m+1]por[N] entrada camada 1
    y1 = np.tanh(w1 @ x1)       # saida das funcoes de ativacao para todos os neuronios e para todo o lote. y1:[Nl(1)]por[N]
    x2 = np.vstack([y1 , np.ones(N)])  # ones representa os bias. x2:[Nl[0]+1]por[N]
    # y2 = np.tanh(w2 @ x2)     # saida COM funcao de ativacao
    y2 = w2 @ x2           # saida SEM funcao de ativacao. y2:[Nl[1]]por[N]
    erro = y2-S            # erro na camada de saida. erro:[Nl[1]]por[N]
    
    Y = y2
    # Calcula erro quadratico medio Ew = J_N
    verro = np.reshape(erro,N*Nl[1])     # vetor de erro, verro:[N*Nl[1]]
    Ew = (0.5/N)*(verro @ verro)            # Ew:[1]x[1], escalar
    return Y,Ew

#%%
def decisorwine(Y,S,S1,S2,S3):

    N = Y.shape[1]
    
    contador1,acertos1 = 0,0    # contadores classe 1
    contador2,acertos2 = 0,0    # contadores classe 2
    contador3,acertos3 = 0,0    # contadores classe 3
    
    decisao = np.zeros(N)
    for ii in range(N):
        if Y[0,ii]<((S1+S2)/2): # <-0.5          # hard limiter 1
            decisao[ii] = 1
            if abs(S[ii]-S1)<1e-12:
                acertos1 = acertos1 + 1
                contador1 = contador1 + 1
            else:
                contador1 = contador1 + 1
        elif (Y[0,ii]>=((S1+S2)/2) and Y[0,ii]<=((S2+S3)/2)): # >=-0.5 <=0.5   % hard limiter 2
            decisao[ii] = 2
            if abs(S[ii]-S2)<1e-12:
                acertos2 = acertos2 + 1
                contador2 = contador2 + 1
            else:
                contador2 = contador2 + 1
        elif Y[0,ii]>((S2+S3)/2):  # >0.5            # hard limiter 3
            decisao[ii] = 3
            if abs(S[ii]-S3)<1e-12:
                acertos3 = acertos3 + 1
                contador3 = contador3 + 1
            else:
                contador3 = contador3 + 1
    
    pacertos = np.array([acertos1/contador1,
                            acertos2/contador2,
                            acertos3/contador3])  # porcentagens de acertos das 3 classes
    
    return pacertos

#%%
# Busca em linha pelo metodo da secao aurea
def razao_aurea(vw,dk,intervalo,Tol,X,S,Nl,m,npesos,N):
    raurea = 0.618033988749895 # (-1.+sqrt(5.))/2. # razao aurea = 0.618

    a,b,vlambda,vmi,f_lambda,f_mi = list(),list(),list(),list(),list(),list()
    kk=0
    a.append(intervalo[0]) # = 0, limite inferior do intervalo
    b.append(intervalo[1]) # = alphamax, limite superior do intervalo

    vlambda.append(a[kk] + (1.-raurea)*(b[kk]-a[kk]))
    vmi.append(a[kk] + raurea*(b[kk]-a[kk]))

    f_lambda.append(evalJw(vw+vlambda[kk]*dk,X,S,Nl,m,npesos,N))
    f_mi.append(evalJw(vw+vmi[kk]*dk,X,S,Nl,m,npesos,N))
    
    while (b[kk]-a[kk]) >= Tol:
        if f_lambda[kk] > f_mi[kk]:
            a.append(vlambda[kk])   # a[k+1] = vlambda[k]
            b.append(b[kk])         # b[k+1] = b[k]
            vlambda.append(vmi[kk])
            vmi.append( a[kk+1] + raurea*(b[kk+1]-a[kk+1]))
            f_lambda.append(f_mi[kk])
            f_mi.append(evalJw(vw+vmi[kk+1]*dk,X,S,Nl,m,npesos,N))
            kk = kk + 1
        else:
            a.append(a[kk])         # a[k+1] = a[k]
            b.append(vmi[kk])       # b[k+1] = vmi[k]
            vmi.append(vlambda[kk])
            vlambda.append(a[kk+1] + (1.-raurea)*(b[kk+1]-a[kk+1]))
            f_lambda.append(evalJw(vw+vlambda[kk+1]*dk,X,S,Nl,m,npesos,N))
            f_mi.append(f_lambda[kk])
            kk = kk + 1
        
    return (a[kk] + b[kk])/2. # alphaotimo

#%%
def cria_padroes_predicao_serie(xn_N,M,nstep):
    ''' cria matriz de sinais recebidos dada a ordem M
        de entrada da RNA e o num. de amostras N da serie temporal
        N = num. amostras da serie no aprendizado
        M = dimensao de entrada da RNA, cujos padroes sao formados por janela
        deslizante de tamanho M deslocadas de 1 amostra cada
        Entradas: xn_N = serie temporal
        Saidas: Xn = matriz dos padroes em formato de vetor coluna
                Sn = vetor dos valores a serem preditos (sinal desejado) em 1
                      amostra (passo 1)
        OBS: shift de 1 amostra para captar o padrao seguinte na sequencia
    '''
    N = len(xn_N)
    
    # alocacao de memoria
    Xn = np.zeros((M,N-M-1))  # matriz dos padroes pre-alocada
    Sn = np.zeros(N-M-1)      # vetor dos valores a serem preditos x(n)
    vind = np.arange(M,N-nstep+1)
    for n in vind:    # o ultimo valor da serie eh o ultimo a ser predito
        # nao ha transiente de preenchimento de zeros
        Xn[:,n-M] = np.flip(xn_N[0:n][-M:])
        Sn[n-M] = xn_N[n+nstep-1]
    return Xn,Sn