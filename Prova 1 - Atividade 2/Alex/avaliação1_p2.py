"""
Acadêmico: Alex Seidi Noce
RA: 101207
Diciplina: Sistemas Inteligentes - 6694
Data: 30 de novembro de 2020
"""
import numpy as np
import matplotlib.pyplot as plt
import lib3_6694 as mylib
import preproc_predicao_Ax as pred_Ax
import scipy.io
import os
np.seterr(divide='ignore', invalid='ignore')

os.system('clear')
mat = scipy.io.loadmat('dados_Ax.mat')
data = mat['sinal_Ax'].flatten()
Data = mylib.mlpdata(data,'acc')
Data.normalize()
X,Xv,Xt = Data.split()
Xp,Yp = mylib.prediction(X,52,2)
Xvp,Yvp = mylib.prediction(Xv,52,2)
Xtp,Ytp = mylib.prediction(Xt,52,2)
print('Número de amostras para treinamento = {}'.format(Xp.shape[1]))
print('Número de amostras para validação = {}'.format(Xvp.shape[1]))
print('Número de amostras para teste = {}'.format(Xtp.shape[1]))
print('Dados da rede:\n-52 entradas\n-layer 1: 10\n-1 saída\n')
#A biblioteca caeita N camadas (layers), porém todas ,infelizmente, usam a tangente hiperbólica
#Dados
wmax = 2
T_g = 1e-3
T_x =1e-6
#-------------------------------------------------------------------------------
"""
i)Fazer treinamento em btach, mini batch e online
"""
"""
MPL(nl,mb=0)
nl: lista dde layers e neurônios, 
    nl[0]: número de entradas
    nl[l]: número de saídas
mb: número de amostras em um mini bach 
"""
#-------------------------------------------------------------------------------
"""
#Treino por barch
Network = mylib.MLP([52,10,1]) #criação da mlp, 52 entradas, 10 neuronios na camada intermediárias e 1 saída
print('\nIniciando o treinamento em BATCH\nÉpocas = 500\n')
EQM_T,EQM_V = Network.train(Xp,Yp,Xvp,Yvp,500,wmax,'batch',T_g,T_x) #treinamento e validação
y,e0 = Network.test(Xtp,Ytp) # teste da mlp
plot0 = mylib.plotEVO(0,EQM_T,'b',EQM_V,'r','Evolução (Batch)') #plot EQM entre traino e validação
plot10 = mylib.plotPrediction(10,Ytp,y.flatten(),'Predição (batch)') #plot da predição dos dados de teste
"""
"""
.train(x,y,xv,yv,epocas,wmax,typ1,t_g,t_x,typ2='SDG',shuf=True)
x/y, xv,yv: entradas e saídas para treinamento e validação
epocas: número de épocas
wmax: máximo valor em módulo dos pesos
typ1: tipo de treinamento (batch,mini batch e online)
t_g: tolerância do gradiente
t_x: tolerância para alpha ótimo
typ2: tipo de método (SDG,CG,BFGS)
    OBS: no uso do CG ou BFGS o número de interações foi limitada a três para agilizar o processo
shuf: embaralhar entradas
"""

#-------------------------------------------------------------------------------
"""
#Treino por mini batch
Network = mylib.MLP([52,10,1],20)
print('\nIniciando o treinamento em MINI BATCH (20)\nÉpocas = 500\n')
EQM_T,EQM_V = Network.train(Xp,Yp,Xvp,Yvp,500,wmax,'mini batch',T_g,T_x)
y,e1 = Network.test(Xtp,Ytp)
plot1 = mylib.plotEVO(1,EQM_T,'b',EQM_V,'r','Evolução (Mini Batch - 20)')
plot11 = mylib.plotPrediction(11,Ytp,y.flatten(),'Predição (mini batch-20)')
#-------------------------------------------------------------------------------
"""
"""
#Treino online
Network = mylib.MLP([52,10,1])
print('\nIniciando o treinamento ONLINE\nÉpocas = 200')
EQM_T,EQM_V = Network.train(Xp,Yp,Xvp,Yvp,200,wmax,'online',T_g,T_x)
y,e2 = Network.test(Xtp,Ytp)
plot2 = mylib.plotEVO(2,EQM_T,'b',EQM_V,'r','Evolução (Online)')
plot22 = mylib.plotPrediction(22,Ytp,y.flatten(),'Predição (online)')

print('Erro de teste:\nErro para o teste em batch: {}\nErro para o test em mini batch: {}'.format(e0,e1))
print('Erro para o teste online: {}'.format(e2))

plt.show()
"""
#-------------------------------------------------------------------------------
"""
ii)Usa o BFGS ou Gradiente conjugado
-Analisar do desempenho usando um batch e mini batch de 20 amostras
-Fazer um predição para 5, 10 e 52 amostras, plotar o EQM x Épocas
"""
#Dados para padrões 5,10 e 52
#treino
X5,Y5 = mylib.prediction(X,52,5) #5 passos
X10,Y10 = mylib.prediction(X,52,10) #10 passos
X52,Y52 = mylib.prediction(X,52,52) #52 passos
#validação
Xv5,Yv5 = mylib.prediction(Xv,52,5) #5 passos
Xv10,Yv10 = mylib.prediction(Xv,52,10) #10 passos
Xv52,Yv52 = mylib.prediction(Xv,52,52) #52 passos
#teste
Xt5,Yt5 = mylib.prediction(Xt,52,5) #5 passos
Xt10,Yt10 = mylib.prediction(Xt,52,10) #10 passos
Xt52,Yt52 = mylib.prediction(Xt,52,52) #52 passos
#----------------------------------------------------------------------------------
epoca=500
#5 passos
# Network1 = mylib.MLP([52,10,1]) ##batch
# print('treinamento em batch (5 passos)')
# EQM_T1,EQM_V1 = Network1.train(X5,Y5,Xv5,Yv5,epoca,wmax,'batch',T_g,T_x,Typ2='BFGS')
# Network2 = mylib.MLP([52,10,1],20) ##mini batch (20 amostras)
# print('\ntreinamento em mini batch (5 passos)')
# EQM_T2,EQM_V2 = Network2.train(X5,Y5,Xv5,Yv5,epoca,wmax,'mini batch',T_g,T_x,Typ2='BFGS')
# fig0 = mylib.plotEVO2(EQM_T1,EQM_V1,EQM_T2,EQM_V2,'Evolução (5 passos)\nbatch','Mini batch')
#---------------------------------------------------------------------------------------------
#10 passos
# Network1 = mylib.MLP([52,10,1]) ##batch
# print('\ntreinamento em batch (10 passos)')
# EQM_T1,EQM_V1 = Network1.train(X10,Y10,Xv10,Yv10,epoca,wmax,'batch',T_g,T_x,Typ2='BFGS')
# Network2 = mylib.MLP([52,10,1],20) ##mini batch (20 amostras)
# print('\ntreinamento em mini batch (10 passos)')
# EQM_T2,EQM_V2 = Network2.train(X10,Y10,Xv10,Yv10,epoca,wmax,'mini batch',T_g,T_x,Typ2='BFGS')
# frig1 = mylib.plotEVO2(EQM_T1,EQM_V1,EQM_T2,EQM_V2,'Evolução (10 passos)\nbatch','Mini batch')
#-----------------------------------------------------------------------------------------------
#52 passos
"""
    OBS: para o mini batch de 52 passos algumas vezes aqui em casa o treinameno parou no meio, antes de completar as épocas, e outras completou normalmente.
    eu não achei o problema, já que o erro só ocorre neesa chamada.

"""
Network1 = mylib.MLP([52,10,1]) ##batch
print('\ntreinamento em batch (52 passos)')
EQM_T1,EQM_V1 = Network1.train(X52,Y52,Xv52,Yv52,epoca,wmax,'batch',T_g,T_x,Typ2='CG')
Network2 = mylib.MLP([52,10,1],20) ##mini batch (20 amostras)
print('\ntreinamento em mini batch (52 passos)')
EQM_T2,EQM_V2 = Network2.train(X52,Y52,Xv52,Yv52,epoca,wmax,'mini batch',T_g,T_x,Typ2='CG')
fig2 = mylib.plotEVO2(EQM_T1,EQM_V1,EQM_T2,EQM_V2,'Evolução (52 passos)\nbatch','Mini batch')
plt.show()
