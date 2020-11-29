# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:27:10 2020

  Atividade 2 Sistemas Inteligentes
@author: Hugo, Lucas, Caio, Bia
"""
import predicting
batchSize = int(input("Please enter the batche sample size:\n"))
predicting.minibatch(2, batchSize)

predicting.minibatch(5)
predicting.minibatch(10)
predicting.minibatch(52)