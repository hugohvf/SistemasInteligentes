% Mutacao rand/1 do DE original
% Input: Pop - populacao dos individuos (target vectors)
%        F - amplification constant
% Output: mPop - populacao mutada
function mPop = mutacao_rand_1(Pop,F,range)

% verifica dimensoes
[Np, dim] = size(Pop);
% Pre-alocacao de memoria
mPop = zeros(Np,dim);
idx = zeros(Np,6);   % matriz dos indices r1, r2 e r3

% forme os indices r1, r2 e r3 aleatorios e mutuamente distintos para toda
% a populacao
if Np < 6
    error('Np deve ser maior ou igual a 6.');
end
for ii=1:Np
    idx(ii,:) = randperm(Np,6);  % OBS: restricao de Np >=6
end

for ii=1:Np,
    % checar se os indices r1,r2 e r3 nao sao iguais a ii
    ind = idx(ii,1:3); II = ind == ii;
    if any(II),
        % use indices que com certeza nao sao iguais a ii
        ind = idx(ii,4:6);
    end
    % realize a mutacao do individuo
    mPop(ii,:) = Pop(ind(1),:) + F*(Pop(ind(2),:)-Pop(ind(3),:));
end

% verificacao se alguma componente violou o seu range
for jj=1:dim,
    % intervalo desta componente
    idxs = mPop(:,jj) < range(jj,1) | mPop(:,jj) > range(jj,2);
    if any(idxs),
        % retorne ao valor original do target vector
        mPop(idxs,jj) = Pop(idxs,jj);
    end
end