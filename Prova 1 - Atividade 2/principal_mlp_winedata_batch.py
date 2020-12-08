# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:18:17 2020

Arquivo principal de uso de uma MLP para classificação de
tipos de vinho, usando treinamento em batelada (com todos)
os padrões de uma vez para encontrar o gradiente.

@author: Rafael
"""
import numpy as np
import matplotlib.pyplot as plt

import mlp_6694 as mymlp
import optimize_6694 as myopt
flagembaralha = 1   # 1 = embaralha, 0 = nao embaralha

#%% Dados de treinamento, validacao e teste
#-------------- Abre o arquivo dos dados de treinamento: wine --------
import preproc_winedata as winedata # load modulo winedata
(X,S,Xval,Sval,Xtest,Stest) = winedata.preproc() # gera dados escalonados da wine recognition

#X (matriz de entrada [N]x[m]) e S (vetor de saida [N]])
(N,Natr) = np.shape(X)
print('Numero de padroes para treinamento = {}'.format(N))

Nval = Xval.shape[0]
print('Numero de padroes para validacao = {}'.format(Nval))

Ntest = Xtest.shape[0]
print('Numero de padroes para teste = {}'.format(Ntest))

#%% Especificacoes, pre-alocacao de memoria e inicializacao
#------------------------------------------------------------------------
# transpor para executar o fluxo de processamento em broadcast (forma matricial)
X, S = X.T, S.T             # transpor para ter os padroes por coluna
Xval, Sval = Xval.T, Sval.T
Xtest, Stest = Xtest.T, Stest.T

Nepocas = 1000    # Num. maximo de epocas para o treinamento
Tolx = 1e-6       # tolerancia para a diferenca dos parametros de busca
Tolgrad = 1e-3    # tolerancia para a norma do gradiente
m = Natr          # Dimensao do vetor de entrada
Nl = [10,1]       # vetor contendo o numero de neuronios na camada intermediaria e na camada de saida [in,out]
w_scale = 2.      # modulo do valor maximo e minimo dos pesos

#--------- pre-alocacao de memoria --------------------------------------
EQM_T_Grad = np.zeros(Nepocas+1)     # erro quadratico do treinamento
EQM_V_Grad = np.zeros(Nepocas+1)     # erro quadratico da validacao
#------------------------------------------------------------------------

#-------------- Inicializacao dos pesos da rede neural -------------------
(w1,w2) = mymlp.inicia_pesos(Nl,m)    # w1:[Nl[0]]por[m+1] w2:[Nl[1]x[Nl[0]+1]
#-------------------------------------------------------------------------
npesos = Nl[0]*(m+1)+Nl[1]*(Nl[0]+1)
print('Numero de pesos na rede neural = {}'.format(npesos))
wmax = w_scale*np.ones(npesos)
wmin = -wmax

k = 0
dEw = np.ones(npesos) # para entrar no loop do treinamento
#%% loop das epocas de treinamento e validacao
while np.linalg.norm(dEw) >= Tolgrad and k <= Nepocas:
    if flagembaralha == 1:
      (X,S) = mymlp.embaralha(X,S)
      (Xval,Sval) = mymlp.embaralha(Xval,Sval)

    # forward and backpropagation (erro quad. e derivada primeira)
    (Ew,dEw) = mymlp.forwardprop_and_backprop_batch(X,S,w1,w2,Nl,m,N)
    EQM_T_Grad[k] = Ew.copy()
    # direcao de atualizacao pelo metodo do gradiente
    d = -dEw/np.linalg.norm(dEw)    # dEw normalizado
    # vetor w, vw:[n[0]*(m+1)+n[1]*(n[0]+1)]x[1]
    vw = np.block([np.reshape(w1,Nl[0]*(m+1)),np.reshape(w2,Nl[1]*(Nl[0]+1))])    
    
    # atualizacao dos pesos pelo metodo do gradiente com busca
    # unidimensional pelo passo
    alphamax = myopt.calc_alpha_max(vw,d,wmin,wmax)
    intervalo = [0,alphamax]
    alphaotimo = mymlp.razao_aurea(vw,d,intervalo,Tolx,X,S,Nl,m,npesos,N)
    vw = vw + alphaotimo*d
    
    # reconstrucao das matrizes w1 e w2 a partir de vw
    w1 = np.reshape(vw[0:Nl[0]*(m+1)],(Nl[0],m+1))
    w2 = np.reshape(vw[npesos-Nl[1]*(Nl[0])-1:npesos],(Nl[1],Nl[0]+1))
    
    # aplica (propaga/processa) conjunto de validacao
    (_,EQM_val) = mymlp.processa(Xval,Sval,w1,w2,Nl,Nval)
    EQM_V_Grad[k] = EQM_val.copy()    # armazena erro quadratico medio da validacao
    
    # atualiza epoca
    k = k + 1
#%%
EQM_T_Grad = EQM_T_Grad[0:k] # seleciona epocas ocorridas apenas
EQM_V_Grad = EQM_V_Grad[0:k]

#%% Teste
(Y,ErroQ) = mymlp.processa(Xtest,Stest,w1,w2,Nl,Ntest)

#--------- Aplicacao do decisor ------------
S1,S2,S3 = -1.,0.,+1.  # valores de saida para identificar as classes 1, 2 e 3.
pacertos = mymlp.decisorwine(Y,Stest,S1,S2,S3)

#%% Resultados
print('Porcentagem de acertos da Classe 1 = {} %'.format(100*pacertos[0]))
print('Porcentagem de acertos da Classe 2 = {} %'.format(100*pacertos[1]))
print('Porcentagem de acertos da Classe 3 = {} %'.format(100*pacertos[2]))

n_epocas = np.arange(k)

plt.figure()
plt.plot(n_epocas,EQM_T_Grad,'b',label = "Treinamento")
plt.plot(n_epocas,EQM_V_Grad,'r',label = "Validação")
plt.legend()
plt.title("Evolução do EQM")
plt.xlabel("Época (iteração)")
plt.ylabel("EQM")
