% Crossover binomial 
% Input: Pop - população de individuos
%       mPop - população mutada
%       CR - taxa de crossover
% Output: U - trial vectors

function U = crossover_bin(Pop,mPop,CR)
% verifica dimensoes
[Np,dim] = size(Pop);
% pre-alocacao de memoria para os trial vectors
U = Pop;

% Trial vectors
auxdim = 1:dim;
for kk=1:Np,
    % cond1 sorteia rj para todas as componentes 
    cond1 = rand(1,dim);
    % cond2 sorteia um indice e compara com o indice das compnentes
    Ii = ones(1,dim)*randi(dim); cond2 = Ii == auxdim; 
    % quais compnentes devem vir do individuo mutado?
    II = (cond1 <= CR | cond2);
    % atribuir componentes do vetor mutado ao trial
    U(kk,II) = mPop(kk,auxdim(II));
end