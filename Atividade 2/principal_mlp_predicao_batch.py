# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:27:10 2020

Arquivo principal de uso de uma MLP para predição em uma
série temporal de dados de um acelerômetro montado no peito
de uma pessoa.
A base de dados é esta aqui:
    "Activity Recognition from single chest-mounted accelerometer"
    https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer

A aceleraçã no eixo x de uma atividade praticada por um dos
 participantes foi selecionado para servir de conjunto de
 dados neste exemplo. Estes dados "crus" passaram por um controle
 de qualidade e foram pré-processados por filtros IIR passa baixas
 do tipo Butterworth de 4a ordem. 
Foi empregado o treinamento em batelada (com todos)
os padrões de uma vez para encontrar o gradiente.

@author: Rafael
"""

import numpy as np
import matplotlib.pyplot as plt

import mlp_6694 as mymlp
import optimize_6694 as myopt
flagembaralha = 1   # 1 = embaralha, 0 = nao embaralha

m = 52      #2*52     # dimensao da entrada
nstep = 2   # passo da predicao (num. de amostras para frente)

import preproc_predicao_Ax as pred_Ax # load modulo winedata
(x,xval,xtest) = pred_Ax.preproc() # gera dados escalonados da wine recognition

Ntrain,Nval,Ntest = x.size,xval.size,xtest.size

ntrain = np.arange(1,Ntrain+1)
nval = np.arange(Ntrain+1,Ntrain+Nval+1)
ntest = np.arange(Ntrain+Nval+1,Ntrain+Nval+Ntest+1)
plt.figure()
plt.plot(ntrain,x,'b',label = "Treinamento")
plt.plot(nval,xval,'r',label = "Validação")
plt.plot(ntest,xtest,'g',label = "Teste")
plt.legend()
plt.title("Série temporal: treinamento+validação+teste")
plt.xlabel("Amostra no tempo")
plt.ylabel("Aceleração normalizada Ax")

(X,S) = mymlp.cria_padroes_predicao_serie(x,m,nstep)
#X (matriz de entrada [m]x[N]) e S (vetor de saida [N]])
N = X.shape[1]
print('Numero de padroes para treinamento = {}'.format(N))

(Xval,Sval) = mymlp.cria_padroes_predicao_serie(xval,m,nstep)
Nval = Xval.shape[1]
print('Numero de padroes para validacao = {}'.format(Nval))

(Xtest,Stest) = mymlp.cria_padroes_predicao_serie(xtest,m,nstep)
Ntest = Xtest.shape[1]
Npred = Ntest
print('Numero de padroes para teste = {}'.format(Ntest))

#%% Especificacoes, pre-alocacao de memoria e inicializacao
#------------------------------------------------------------------------

Nepocas = 500    # Num. maximo de epocas para o treinamento
Tolx = 1e-6       # tolerancia para a diferenca dos parametros de busca
Tolgrad = 1e-3    # tolerancia para a norma do gradiente
Nl = [5,1]       # vetor contendo o numero de neuronios na camada intermediaria e na camada de saida [in,out]
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
    
    # imprime status do treinamento
    if k % 10 == 0:
        print('Epoca {}, EQM_T = {:.5f}, EQM_V = {:.5f}'.format(k,
              EQM_T_Grad[k],EQM_V_Grad[k]))
    # atualiza epoca
    k = k + 1
    
#%%
EQM_T_Grad = EQM_T_Grad[0:k] # seleciona epocas ocorridas apenas
EQM_V_Grad = EQM_V_Grad[0:k]

#%% Teste
(Y,ErroQ) = mymlp.processa(Xtest,Stest,w1,w2,Nl,Ntest)
Y = Y.flatten()
#%% Resultados
n_epocas = np.arange(k)

plt.figure()
plt.plot(n_epocas,EQM_T_Grad,'b',label = "Treinamento")
plt.plot(n_epocas,EQM_V_Grad,'r',label = "Validação")
plt.legend()
plt.title("Evolução do EQM")
plt.xlabel("Época (iteração)")
plt.ylabel("EQM")

N = Ntrain+Nval+Ntest
npred = np.arange(N-Npred+nstep,N+nstep)
plt.figure()
plt.plot(npred,Stest,'k',label = "Série real")
plt.plot(npred,Y,'g', label = 'Série predita')
plt.legend()
plt.title('Predição')
plt.xlabel('amostra')
plt.ylabel('Ax normalizado')
'''
y = []
eixo_x = []
for nn in range(Npred):
    y.append(Y[nn])
    eixo_x.append(nn+N-Npred+nstep)
    plt.plot(npred,Stest,'k',label = "Série real")
    plt.plot(eixo_x,y,'g', label = "Série predita")
    plt.plot(nn+N-Npred+nstep,Y[nn],'og')
    #plt.legend()
    plt.title('Predição')
    plt.xlabel('amostra')
    plt.ylabel('Ax normalizado')
    plt.draw()
    plt.pause(0.01)
    #time.sleep(0.1)
'''