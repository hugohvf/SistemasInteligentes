# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:27:10 2020

  Atividade 2 Sistemas Inteligentes
@author: Hugo, Lucas, Caio, Bia
"""
import principal_mlp_predicao_minibatch as minibatch
import principal_mlp_predicao_batch as batch

batchSize = int(input("Please enter the batche sample size:\n"))
minibatch.predict(2, batchSize)

batch.predict(5)
minibatch.predict(5)

batch.predict(10)
minibatch.predict(10)

batch.predict(52)
minibatch.predict(52)
