% atualiza melhor vizinho de cada celula
%Input: Pop - populacao de particulas
%       fPop - vetor de fitness
%       Bn - melhor vizinho de cada particula
%       fBn - fitness dos melhores vizinhos
%       Index - matriz com os indices dos vizinhos ([N]x[2] - topologia em anel)
%Output: Bn,fBn - melhores vizinhos atualizados

function [Bn,fBn] = best_neighbor(Pop,fPop,Bn,fBn,Index)

% verifica dimensoes
N = size(Pop,1); 

for i=1:N,
    % melhor vizinho e salva seu fitness
    [vbest,idx] = max(fPop(Index(i,:)));
    if vbest > fBn(i), 
        if idx == 1,
            % o melhor eh o primeiro vizinho ("direita")
            Bn(i,:) = Pop(Index(i,1),:); fBn(i) = fPop(Index(i,1));
        else
            % o melhor eh o segundo vizinho ("esquerda")
            Bn(i,:) = Pop(Index(i,2),:); fBn(i) = fPop(Index(i,2));
        end
    end
end