% criterio de Selecao guloso (somento os melhores sobrevivem)
% Input: Pop - população de individuos
%        fPop - fitness da população
%        U - trial vectors (população mutada e com crossover)
%        fU - fitness dos trial vectors
% Output: Pop - nova população
%         fPop - fitness da nova população

function [Pop,fPop] = selecao_gulosa(Pop,fPop,U,fU)
    
sel = (fU > fPop);
Pop(sel,:) = U(sel,:);
fPop(sel) = fU(sel);