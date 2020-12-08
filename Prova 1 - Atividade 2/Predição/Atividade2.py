# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:21:37 2020

Atividade 2 Sistemas Inteligentes

@author: Hugo, Lucas, Caio, Bia
"""
import mlp_predicao as predicao
import matplotlib.pyplot as plt

method = "BFGS"

batchSize = int(input("Insira o tamanho do batch para o treinamento por minibatch:\n"))
print('\nPredição do batch e minibatch com tamanho de {}:'.format(batchSize))
(EQM_T_Grad1, EQM_V_Grad1, Stest1, Y1) = predicao.run("batch", 2, method)
(EQM_T_Grad2, EQM_V_Grad2, Stest2, Y2) = predicao.run("minibatch", 2, method, batchSize)

fig2 = predicao.plotEvolution('Evolução com minibatch de tamanho {}\nTipo batch'.format(batchSize), 'Tipo minibatch', EQM_T_Grad1, EQM_V_Grad1, EQM_T_Grad2, EQM_V_Grad2)
plt.show()
fig1 = predicao.plotPrediction('Predição com minibatch de tamanho {}\nTipo batch'.format(batchSize), 'Tipo minibatch', Stest1, Y1, Stest2, Y2)
plt.show()

print('\nTreinamento batch com ntep 5:')
(EQM_T_Grad1, EQM_V_Grad1, Stest1, Y1) = predicao.run("batch", 5, method)
print('\nTreinamento minibatch com ntep 5:')
(EQM_T_Grad2, EQM_V_Grad2, Stest2, Y2) = predicao.run("minibatch", 5, method)

fig2 = predicao.plotEvolution('Evolução com 5 passos\nTipo batch', 'Tipo minibatch', EQM_T_Grad1, EQM_V_Grad1, EQM_T_Grad2, EQM_V_Grad2)
plt.show()
fig1 = predicao.plotPrediction('Predição com 5 passos\nTipo batch', 'Tipo minibatch', Stest1, Y1, Stest2, Y2)
plt.show()

print('\nTreinamento batch com ntep 10:')
(EQM_T_Grad1, EQM_V_Grad1, Stest1, Y1) = predicao.run("batch", 10, method)
print('\nTreinamento batch com ntep 10:')
(EQM_T_Grad2, EQM_V_Grad2, Stest2, Y2) = predicao.run("minibatch", 10, method)

fig2 = predicao.plotEvolution('Evolução com 10 passos\nTipo batch', 'Tipo minibatch', EQM_T_Grad1, EQM_V_Grad1, EQM_T_Grad2, EQM_V_Grad2)
plt.show()
fig1 = predicao.plotPrediction('Predição com 10 passos\nTipo batch', 'Tipo minibatch', Stest1, Y1, Stest2, Y2)
plt.show()

print('\nTreinamento batch com ntep 52:')
(EQM_T_Grad1, EQM_V_Grad1, Stest1, Y1) = predicao.run("batch", 52, method)
print('\nTreinamento batch com ntep 52:')
(EQM_T_Grad2, EQM_V_Grad2, Stest2, Y2) = predicao.run("minibatch", 52, method)

fig2 = predicao.plotEvolution('Evolução com 52 passos\nTipo batch', 'Tipo minibatch', EQM_T_Grad1, EQM_V_Grad1, EQM_T_Grad2, EQM_V_Grad2)
plt.show()
fig1 = predicao.plotPrediction('Predição com 52 passos\nTipo batch', 'Tipo minibatch', Stest1, Y1, Stest2, Y2)
plt.show()