% Mutação Gaussiana
%Input: Pop - população de individuos
%       prob - probabilidade de cada componente sofrer mutação
%Output: Popm - população com mutação

function Popm = mutacao_gaussiana(Pop, prob, range)

% verifica num. de individuos e dimensao
[Np,dim] = size(Pop);
% pre-alocacao de memoria
Popm = zeros(Np,dim);

for kk=1:dim,
    % vetor de mutacao gaussiana
    Mut = randn(Np,1); Popm(:,kk) = Pop(:,kk) + Mut; 
    % verifique quais individuos não deveriam sofrer mutacao
    [Is] = find(rand(Np,1) > prob); Popm(Is,kk) = Pop(Is,kk);
    % verifique se cada componente nao extrapolou o seu intervalo de busca
    idxs = find(Popm(:,kk) < range(kk,1) | Popm(:,kk) > range(kk,2));
    if ~isempty(idxs),
        % retorne ao seu valor original
        Popm(idxs,kk) = Pop(idxs,kk); 
    end
end

