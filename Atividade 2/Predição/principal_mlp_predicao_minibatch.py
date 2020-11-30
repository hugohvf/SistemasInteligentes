"""

Função de predição de Batelada
Created on Sun Nov  1 15:33:24 2020
@author: Hugo, Lucas, Caio, Bia
"""

import preproc_predicao_Ax as pred_Ax  # load modulo winedata
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import mlp_6694 as mymlp
import optimize_6694 as myopt

def predict(nstep = 2, batchSize = 20):
  """
  batchSize: The amount of data included in each sub-epoch weight change
  nstep: predicting step (number of samples ahead)
  """
  flagembaralha = 1   # 1 = embaralha, 0 = nao embaralha

  m = 52  # 2*52     # dimensao da entrada

  # gera dados escalonados da wine recognition
  (x, xval, xtest) = pred_Ax.preproc()

  Ntrain, Nval, Ntest = x.size, xval.size, xtest.size

  ntrain = np.arange(1, Ntrain+1)
  nval = np.arange(Ntrain+1, Ntrain+Nval+1)
  ntest = np.arange(Ntrain+Nval+1, Ntrain+Nval+Ntest+1)
  plt.figure()
  plt.plot(ntrain, x, 'b', label="Treinamento")
  plt.plot(nval, xval, 'r', label="Validação")
  plt.plot(ntest, xtest, 'g', label="Teste")
  plt.legend()
  plt.title("Série temporal: treinamento+validação+teste")
  plt.xlabel("Amostra no tempo")
  plt.ylabel("Aceleração normalizada Ax")
  plt.show()

  (X, S) = mymlp.cria_padroes_predicao_serie(x, m, nstep)
  # X (matriz de entrada [m]x[N]) e S (vetor de saida [N]])
  N = X.shape[1]
  print('Numero de padroes para treinamento = {}'.format(N))

  (Xval, Sval) = mymlp.cria_padroes_predicao_serie(xval, m, nstep)
  Nval = Xval.shape[1]
  print('Numero de padroes para validacao = {}'.format(Nval))

  (Xtest, Stest) = mymlp.cria_padroes_predicao_serie(xtest, m, nstep)
  Ntest = Xtest.shape[1]
  Npred = Ntest
  print('Numero de padroes para teste = {}'.format(Ntest))

  # %% Especificacoes, pre-alocacao de memoria e inicializacao
  # ------------------------------------------------------------------------

  Nepocas = 500    # Num. maximo de epocas para o treinamento
  Tolx = 1e-6       # tolerancia para a diferenca dos parametros de busca
  Tolgrad = 1e-3    # tolerancia para a norma do gradiente
  # vetor contendo o numero de neuronios na camada intermediaria e na camada de saida [in,out]
  Nl = [5, 1]
  w_scale = 2.      # modulo do valor maximo e minimo dos pesos

  # --------- pre-alocacao de memoria --------------------------------------
  EQM_T_Grad = np.zeros(Nepocas+1)     # erro quadratico do treinamento
  EQM_V_Grad = np.zeros(Nepocas+1)     # erro quadratico da validacao
  # ------------------------------------------------------------------------

  # -------------- Inicializacao dos pesos da rede neural -------------------
  # w1:[Nl[0]]por[m+1] w2:[Nl[1]x[Nl[0]+1]
  (w1, w2) = mymlp.inicia_pesos(Nl, m)
  # -------------------------------------------------------------------------
  npesos = Nl[0]*(m+1)+Nl[1]*(Nl[0]+1)
  print('Numero de pesos na rede neural = {}'.format(npesos))
  wmax = w_scale*np.ones(npesos)
  wmin = -wmax

  k = 0
  dEw = np.ones(npesos)  # para entrar no loop do treinamento

  batchIterations = N // batchSize
  firstBatch = X[:, 0:batchSize]
  print(f'You will use {batchIterations} batches with {batchSize} samples')

  # %% loop das epocas de treinamento e validacao
  while np.linalg.norm(dEw) >= Tolgrad and k <= Nepocas:
      for miniBatch in range(batchIterations):
          XminiBatch = X[:, batchSize*miniBatch:batchSize*(miniBatch+1)]
          SminiBatch = S[batchSize*miniBatch:batchSize*(miniBatch+1)]

          if flagembaralha == 1:
              (XminiBatch, SminiBatch) = mymlp.embaralha(XminiBatch, SminiBatch)
              (Xval, Sval) = mymlp.embaralha(Xval, Sval)

          # forward and backpropagation (erro quad. e derivada primeira)
          (Ew, dEw) = mymlp.forwardprop_and_backprop_batch(
              XminiBatch, SminiBatch, w1, w2, Nl, m, batchSize)
          EQM_T_Grad[k] = Ew.copy()

          vw = np.block([w1.flatten(), w2.flatten()])

          w1 = np.reshape(vw[0:Nl[0] * (m + 1)], (Nl[0], m + 1))
          w2 = np.reshape(vw[npesos - Nl[1] * (Nl[0]) - 1:npesos], (Nl[1], Nl[0] + 1))

          optimized = optimize.minimize(cost, vw, args=(XminiBatch, SminiBatch, batchSize, Nl, m, npesos), method="CG", jac=True, options = {'maxiter': 5})
          vw = optimized.x

          # reconstrucao das matrizes w1 e w2 a partir de vw
          w1 = np.reshape(vw[0:Nl[0]*(m+1)], (Nl[0], m+1))
          w2 = np.reshape(vw[npesos-Nl[1]*(Nl[0])-1:npesos], (Nl[1], Nl[0]+1))

          # aplica (propaga/processa) conjunto de validacao
          (_, EQM_val) = mymlp.processa(Xval, Sval, w1, w2, Nl, Nval)
          # armazena erro quadratico medio da validacao
          EQM_V_Grad[k] = EQM_val.copy()


      # imprime status do treinamento
      if k % 10 == 0:
          print('Epoca {}, EQM_T = {:.5f}, EQM_V = {:.5f}'.format(k,
                                                                  EQM_T_Grad[k], EQM_V_Grad[k]))
      # atualiza epoca
      k = k + 1

  # %%
  EQM_T_Grad = EQM_T_Grad[0:k]  # seleciona epocas ocorridas apenas
  EQM_V_Grad = EQM_V_Grad[0:k]

  # %% Teste
  (Y, ErroQ) = mymlp.processa(Xtest, Stest, w1, w2, Nl, Ntest)
  Y = Y.flatten()
  # %% Resultados
  n_epocas = np.arange(k)

  plt.figure()
  plt.plot(n_epocas, EQM_T_Grad, 'b', label="Treinamento")
  plt.plot(n_epocas, EQM_V_Grad, 'r', label="Validação")
  plt.legend()
  plt.title("Evolução do EQM")
  plt.xlabel("Época (iteração)")
  plt.ylabel("EQM")
  plt.show()

  N = Ntrain+Nval+Ntest
  npred = np.arange(N-Npred+nstep, N+nstep)
  plt.figure()
  plt.plot(npred, Stest, 'k', label="Série real")
  plt.plot(npred, Y, 'g', label='Série predita')
  plt.legend()
  plt.title('Predição')
  plt.xlabel('amostra')
  plt.ylabel('Ax normalizado')
  plt.show()

def cost(vw, X, S, batchSize, Nl, m, npesos):
  w1 = np.reshape(vw[0:Nl[0] * (m + 1)], (Nl[0], m + 1))
  w2 = np.reshape(vw[npesos - Nl[1] * (Nl[0]) - 1:npesos], (Nl[1], Nl[0] + 1))
  (Ew, dEw) = mymlp.forwardprop_and_backprop_batch(
    X, S, w1, w2, Nl, m, batchSize)
  dEw = dEw / np.linalg.norm(dEw)
  return Ew, dEw
