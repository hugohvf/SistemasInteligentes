% criterio de Selecao guloso (somento os melhores sobrevivem)
% Input: Pop - popula��o de individuos
%        fPop - fitness da popula��o
%        U - trial vectors (popula��o mutada e com crossover)
%        fU - fitness dos trial vectors
% Output: Pop - nova popula��o
%         fPop - fitness da nova popula��o

function [Pop,fPop] = selecao_gulosa(Pop,fPop,U,fU)
    
sel = (fU > fPop);
Pop(sel,:) = U(sel,:);
fPop(sel) = fU(sel);