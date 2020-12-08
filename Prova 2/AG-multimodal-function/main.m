% 6694 - Sistemas Inteligentes - Eng. Elétrica/UEM
% Prof. Rafael Krummenauer
% 
% Implementação: Algoritmo Genético
% Autoria: Rafael Krummenauer
% Data: 13/jul/2016
% Nota: programas baseados na dissertação de mestrado e de códigos
%       do prof. Levy Bocatto, UNICAMP.
clear all; close all;
clc;
%% Ajuste dos parâmetros

Np = 40;             % num. de individuos na populacao
max_it = 100;         % maximo num. de iteracoes
dim = 2;            % dimensao do espaco de busca
pc = 0.5; pm = 0.3; % probabilidades de crossover e de mutacao
q = 3;              % numero de individuos em cada torneio (selecao)

%% Define da função custo e plota a superficie
f = '1 * x .* sin(4 * pi .* x) - 1 * y.* sin(4 * pi .* y + pi) + 1'; 
xmin = -1; xmax = 2; ymin = -1; ymax = 2;
range = [xmin xmax; ymin ymax];   % intervalo de busca de cada variavel
[x,y] = meshgrid(xmin:0.04:xmax,ymin:0.04:ymax); PLOT.x = x; PLOT.y = y;
PLOT.z = eval(f);

%% Executa o AG
RESULT = AG(Np,dim,range,pc,pm,max_it,q,PLOT);

%% Plota as curvas de fitness
it_vec = 1:max_it;
plot(it_vec,RESULT.fmean,'b',it_vec,RESULT.fbest,'r');
legend('fitness médio','fitness do melhor indivíduo');
ylabel('Evolução do fitness'); xlabel('Geração (iteração)')
figure; imprime(PLOT.x,PLOT.y,PLOT.z,RESULT.Pop(:,1),RESULT.Pop(:,2),log(RESULT.fPop),1,1);