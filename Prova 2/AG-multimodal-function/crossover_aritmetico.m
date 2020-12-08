% Crossover aritmético
%Input: x1,x2 - individuos a serem recombinados [1]x[dim]
%Output: f1,f2 - filhos/descendentes gerados

function [f1,f2] = crossover_aritmetico(x1,x2)

% verifica o num. de componentes no individuo
dim = length(x1);
% escolhe aleatoriamente a ponderacao para cada componente (separadamente)
alpha = rand(1,dim);
% gera os descendentes/filhos
f1 = alpha.*x1 + (1-alpha).*x2; f2 = (1-alpha).*x1 + alpha.*x2;
