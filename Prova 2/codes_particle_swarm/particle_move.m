% atualiza a posicao de cada particula na direcao de V
%Input: Pop - populacao de particulas
%       V - vetores de velocidade
%       range - [x1min x1max;x2min x2max;...] [dim]x[2]
%Output: newPop,fnewPop - populacao atualizada e seu vetor de fitness

function [newPop,fnewPop] = particle_move(Pop,V,range)

% avalia dimensao das particulas
dim = size(Pop,2);
% move cada particula
newPop = Pop + V;
% verifica se houve violacao de intervalo em cada dimensao
for kk=1:dim,
    % intervalo desta componente
    idxs = newPop(:,kk) < range(kk,1) | newPop(:,kk) > range(kk,2);
    if max(idxs) ~= 0,
        % retorne ao valor original
        newPop(idxs,kk) = Pop(idxs,kk);
    end
end
% novo vetor de fitness
fnewPop = fitness(newPop);
