% Encontra melhor vizinho inicial de cada célula
%Input: Pop - população de partículas
%       fPop - vetor de fitness
%       Index - matriz com os índices dos vizinhos ([N]x[2] - topologia em anel)
%Output: Bn,fBn - melhores vizinho da população e seus fitness

function [Bn,fBn] = best_initial(Pop,fPop,Index)

% verifica dimensoes
[N,dim] = size(Pop); 
% pre-alocacao de memoria para a saida
Bn = zeros(N,dim); fBn = zeros(N,1);

for i=1:N,
    % melhor vizinho e salva seu fitness
    [fBn(i),idx] = max(fPop(Index(i,:)));
    if idx == 1,
        % o melhor eh o primeiro vizinho ("direita")
        Bn(i,:) = Pop(Index(i,1),:); 
    else
        % o melhor eh o segundo vizinho ("esquerda")
        Bn(i,:) = Pop(Index(i,2),:); 
    end
end