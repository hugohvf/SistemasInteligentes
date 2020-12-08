% 6694 - Sistemas Inteligentes - Eng. El�trica/UEM
% Prof. Rafael Krummenauer
% 
% Implementa��o: Evolu��o Diferencial
% Autoria: Rafael Krummenauer
% Email: rkrummenauer@gmail.com
% Data: 20/jul/2016
% Nota: programas baseados na disserta��o de mestrado
%       do prof. Levy Bocatto, UNICAMP.
clear all; close all;
clc;
%% Ajuste dos par�metros

Np = 20;        % num. de individuos na populacao
max_it = 50;   % maximo num. de iteracoes
dim = 2;        % dimensao do espaco de busca
F = 0.5;        % passo diferencial
CR = 0.1;       % probabilidade de crossover

%% Define da fun��o custo e plota a superficie
f = '1 * x .* sin(4 * pi .* x) - 1 * y.* sin(4 * pi .* y + pi) + 1'; 
xmin = -1; xmax = 2; ymin = -1; ymax = 2;
range = [xmin xmax; ymin ymax];   % intervalo de busca de cada variavel
[x,y] = meshgrid(xmin:0.04:xmax,ymin:0.04:ymax); PLOT.x = x; PLOT.y = y;
PLOT.z = eval(f);

%% Executa o DE
RESULT = DE(Np,dim,range,F,CR,max_it,PLOT);

%% Plota as curvas de fitness
iter = 0:max_it;
plot(iter,RESULT.fmean,'b',iter,RESULT.fbest,'r');
legend('fitness m�dio','fitness melhor indiv�duo');
xlabel('itera��o'); ylabel('fitness');
figure; imprime(PLOT.x,PLOT.y,PLOT.z,RESULT.Pop(:,1),RESULT.Pop(:,2),log(RESULT.fPop),1,1);