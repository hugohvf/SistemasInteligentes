% 6694 - Sistemas Inteligentes - Eng. Eletrica/UEM
% Prof. Rafael Krummenauer
% 
% Implementacao: Particle Swarm com fator de constricao
% Autoria: Rafael Krummenauer
% Email: rkrummenauer@gmail.com
% Data: 27/jul/2016
% Ultima atualizacao: 03/jul/2018
% Nota: programas baseados na dissertacao de mestrado
%       do prof. Levy Bocatto, UNICAMP.
clear all; close all;

%% Ajuste dos parametros

N = 40;         % num. de particulas
max_it = 200;   % maximo num. iteracoes
dim = 2;        % dimensao do espaco de busca
vmax = 0.1;     % maxima velocidade inicial
csi = 0.729;    % fator de constricao

%% Define da funcao custo e plota a superficie
f = '1 * x .* sin(4 * pi .* x) - 1 * y.* sin(4 * pi .* y + pi) + 1'; 
xmin = -1; xmax = 2; ymin = -1; ymax = 2;
range = [xmin xmax; ymin ymax];   % intervalo de busca de cada variavel
[x,y] = meshgrid(xmin:0.04:xmax,ymin:0.04:ymax); PLOT.x = x; PLOT.y = y;
PLOT.z = eval(f);

%% Executa o PSO
RESULT = PSO_CF(N,dim,range,max_it,vmax,csi,PLOT);

%% Plota as curvas de fitness
iter = 0:max_it;
plot(iter,RESULT.fmean,'b',iter,RESULT.fbest,'r');
legend('fitness médio','fitness melhor indivíduo');
xlabel('iteração'); ylabel('fitness');
figure; imprime(PLOT.x,PLOT.y,PLOT.z,RESULT.Pop(:,1),RESULT.Pop(:,2),log(RESULT.fPop),1,1);