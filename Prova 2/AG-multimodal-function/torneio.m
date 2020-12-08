% Sele��o por torneio - implementa��o tradicional
%Input: Pop - popula��o de individuos
%       fPop - fitness de cada individuo da popula��o
%       q - num. de competidores em cada torneio
%       N - num. de individuos que devem ser selecionados

function [NewPop,newfPop] = torneio(Pop,fPop,q,N)

% verifica o num. de individuos e dimensao do espaco de busca
[Np,dim] = size(Pop);
% pre-alocacao de memoria
NewPop = zeros(N,dim); newfPop = zeros(N,1);

for i=1:N,
    % seleciona aleatoriamente q individuos (com reposicao)
    permNp = randperm(Np); idx = permNp(1:q);
    % encontre o melhor entre os selecionados
    [~,ind] = max(fPop(idx));
    % o vencedor vai para a proxima geracao
    NewPop(i,:) = Pop(idx(ind),:); newfPop(i) = fPop(idx(ind));
end

