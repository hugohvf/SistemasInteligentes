% Gera uma população uniformemente distribuida no espaço de busca
%Input: N - num. de individuos a serem gerados
%       range - [x1min x1max; x2min x2max; ... ] [dim]x[2]
%       dim - dimensao do espaço de busca
%Output:    Pop,fPop - população e o fitness de cada individuo

function [Pop,fPop] = pop_inicial(N,range,dim)

% pre-alocacao de memoria
Pop = zeros(N,dim); 
% inicializacao por dimensao (gene)
for i=1:dim,
    Pop(:,i) = range(i,1) + (range(i,2)-range(i,1))*rand(N,1);
end
% avalie o fitness desta populacao inicial
fPop = fitness(Pop);
