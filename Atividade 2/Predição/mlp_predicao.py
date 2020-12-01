"""
Created on Sun Nov  1 15:33:24 2020

Funções auxiliares para predição

@author: Hugo, Lucas, Caio, Bia
"""

import preproc_predicao_Ax as pred_Ax
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import mlp_6694 as mymlp
import optimize_6694 as myopt


def run(typeoftraining, nstep=2, method=False, batchSize=20, maxiter = 15):
	"""
		typeoftraining : batch or minibatch type
		method: otimized method used
		maxiter: number of iterations the method of otimization will use
		batchSize: The amount of data included in each sub-epoch weight change
		nstep: predicting step (number of samples ahead)

		return: QM_T_Grad, EQM_V_Grad, Stest, Y
	"""
	flagembaralha = 1
	m = 52

	(x, xval, xtest) = pred_Ax.preproc()

	(X, S) = mymlp.cria_padroes_predicao_serie(x, m, nstep)

	N = X.shape[1]
	print('Numero de padroes para treinamento = {}'.format(N))

	(Xval, Sval) = mymlp.cria_padroes_predicao_serie(xval, m, nstep)
	Nval = Xval.shape[1]
	print('Numero de padroes para validacao = {}'.format(Nval))

	(Xtest, Stest) = mymlp.cria_padroes_predicao_serie(xtest, m, nstep)
	Ntest = Xtest.shape[1]
	print('Numero de padroes para teste = {}'.format(Ntest))

	Nepocas = 500
	Tolx = 1e-6
	Tolgrad = 1e-3

	Nl = [5, 1]
	w_scale = 2.

	EQM_T_Grad = np.zeros(Nepocas + 1)
	EQM_V_Grad = np.zeros(Nepocas + 1)

	(w1, w2) = mymlp.inicia_pesos(Nl, m)
	npesos = Nl[0] * (m + 1) + Nl[1] * (Nl[0] + 1)
	print('Numero de pesos na rede neural = {}'.format(npesos))
	wmax = w_scale * np.ones(npesos)
	wmin = -wmax

	k = 0
	dEw = np.ones(npesos)

	numberOfMinibatchesIterations = N // batchSize

	if typeoftraining == 'minibatch':
		print(f'You will use {numberOfMinibatchesIterations} batches with {batchSize} samples')

	while np.linalg.norm(dEw) >= Tolgrad and k <= Nepocas:
		if typeoftraining == 'minibatch':
			for miniBatch in range(numberOfMinibatchesIterations):
				posicaoInicio = batchSize * miniBatch
				posicaoFim = batchSize * (miniBatch + 1)

				XminiBatch = X[:, posicaoInicio:posicaoFim]
				SminiBatch = S[posicaoInicio:posicaoFim]

				if flagembaralha == 1:
					(XminiBatch, SminiBatch) = mymlp.embaralha(XminiBatch, SminiBatch)
					(Xval, Sval) = mymlp.embaralha(Xval, Sval)

				(Ew, dEw) = mymlp.forwardprop_and_backprop_batch(
					XminiBatch, SminiBatch, w1, w2, Nl, m, batchSize)
				EQM_T_Grad[k] = Ew.copy()

				vw = np.block([w1.flatten(), w2.flatten()])

				if method:
					r = optimize.minimize(cost, vw, args=(XminiBatch, SminiBatch, batchSize, Nl, m, npesos), method=method, jac=True, options={'maxiter': maxiter})
					vw = r.x
				else:
					alphamax = myopt.calc_alpha_max(vw, d, wmin, wmax)
					intervalo = [0, alphamax]
					alphaotimo = mymlp.razao_aurea(
						vw, d, intervalo, Tolx, X, S, Nl, m, npesos, N)
					vw = vw + alphaotimo * d

				w1 = np.reshape(vw[0:Nl[0] * (m + 1)], (Nl[0], m + 1))
				w2 = np.reshape(vw[npesos - Nl[1] * (Nl[0]) - 1:npesos], (Nl[1], Nl[0] + 1))

				(_, EQM_val) = mymlp.processa(Xval, Sval, w1, w2, Nl, Nval)
				EQM_V_Grad[k] = EQM_val.copy()

		if typeoftraining == 'batch':
			if flagembaralha == 1:
				(X, S) = mymlp.embaralha(X, S)
				(Xval, Sval) = mymlp.embaralha(Xval, Sval)

			(Ew, dEw) = mymlp.forwardprop_and_backprop_batch(X, S, w1, w2, Nl, m, N)
			EQM_T_Grad[k] = Ew.copy()

			d = -dEw / np.linalg.norm(dEw)
			vw = np.block([np.reshape(w1, Nl[0] * (m + 1)),
			               np.reshape(w2, Nl[1] * (Nl[0] + 1))])

			if method:
				r = optimize.minimize(cost, vw, args=(X, S, N, Nl, m, npesos), method=method,
				                              jac=True, options={'maxiter': maxiter})
				vw = r.x
			else:
				alphamax = myopt.calc_alpha_max(vw, d, wmin, wmax)
				intervalo = [0, alphamax]
				alphaotimo = mymlp.razao_aurea(
					vw, d, intervalo, Tolx, X, S, Nl, m, npesos, N)
				vw = vw + alphaotimo * d

			w1 = np.reshape(vw[0:Nl[0] * (m + 1)], (Nl[0], m + 1))
			w2 = np.reshape(vw[npesos - Nl[1] * (Nl[0]) - 1:npesos], (Nl[1], Nl[0] + 1))

			(_, EQM_val) = mymlp.processa(Xval, Sval, w1, w2, Nl, Nval)
			EQM_V_Grad[k] = EQM_val.copy()

		if k % 10 == 0:
			print('Epoca {}, EQM_T = {:.5f}, EQM_V = {:.5f}'.format(k,
			                                                        EQM_T_Grad[k], EQM_V_Grad[k]))
		k = k + 1

	# %%
	EQM_T_Grad = EQM_T_Grad[0:k]
	EQM_V_Grad = EQM_V_Grad[0:k]

	# %%
	(Y, ErroQ) = mymlp.processa(Xtest, Stest, w1, w2, Nl, Ntest)
	Y = Y.flatten()

	return EQM_T_Grad, EQM_V_Grad, Stest, Y

def plotEvolution(Titulo1, Titulo2, EQM_T_Grad1, EQM_V_Grad1, EQM_T_Grad2, EQM_V_Grad2):
	"""
	Plot the evolution of batch and minibatch
	"""
	fig, axs = plt.subplots(2)
	axs[0].set_title(Titulo1)
	axs[1].set_title(Titulo2)
	t1 = np.arange(len(EQM_T_Grad1))
	t2 = np.arange(len(EQM_T_Grad2))
	axs[0].plot(t1, EQM_T_Grad1, lw=0.5, c='b', label='Treinamento')
	axs[0].plot(t1, EQM_V_Grad1, lw=0.5, c='r', label='Validação')
	axs[1].plot(t2, EQM_T_Grad2, lw=0.5, c='b', label='Treinamento')
	axs[1].plot(t2, EQM_V_Grad2, lw=0.5, c='r', label='Validação')
	plt.legend()
	return fig

def plotPrediction(Titulo1, Titulo2, Stest1, Y1, Stest2, Y2):
	"""
		Plot the prediction of batch and minibatch
	"""
	fig, axs = plt.subplots(2)
	axs[0].set_title(Titulo1)
	axs[1].set_title(Titulo2)
	t1 = np.arange(len(Y1))
	t2 = np.arange(len(Y2))
	axs[0].plot(t1, Stest1, lw=0.5, c='k', label='Série real')
	axs[0].plot(t1, Y1, lw=0.5, c='g', label='Série predita')
	axs[1].plot(t2, Stest2, lw=0.5, c='k', label='Série real')
	axs[1].plot(t2, Y2, lw=0.5, c='g', label='Série predita')
	plt.legend()
	return fig

def cost(vw, X, S, N, Nl, m, npesos):
	"""
		Cost function to be optimized
	"""
	w1 = np.reshape(vw[0:Nl[0] * (m + 1)], (Nl[0], m + 1))
	w2 = np.reshape(vw[npesos - Nl[1] * (Nl[0]) - 1:npesos], (Nl[1], Nl[0] + 1))
	(Ew, dEw) = mymlp.forwardprop_and_backprop_batch(
		X, S, w1, w2, Nl, m, N)
	dEw = dEw / np.linalg.norm(dEw)
	return Ew, dEw
