% DE/rand/1/bin
%Input: N - numero de individuos
%       dim - dimensao do espaço de busca
%       range - [x1min x1max; x2min x2max; ... ] [dim]x[2]
%       F - passo diferencial (mutação)
%       CR - probabilidade de crossover
%       max_it - máximo num. de gerações/iterações
%       PLOT - variaveis de plotagem
%       .x - eixo x
%       .y - eixo y
%       .z - valor da função
%Output: RESULT - estrutura contendo resultados de interesse do DE
%        .Pop - população final
%        .fPop - vetor de fitness final
%        .fbest - evolução do melhor fitness ([1]x[max_it]
%        .fmean - evolução do fitness médio ([1]x[max_it]

function RESULT = DE(Np,dim,range,F,CR,max_it,PLOT)

% gera populacao inicial
[Pop,fPop] = pop_inicial(Np,range,dim);
% pre-alocacao de memoria para armazenamento de melhor fitness e medio
% medio em cada iteracao
fbest = zeros(1,max_it+1); fmean = zeros(1,max_it+1);
% encontre e salve o melhor fitness e o fitness medio desta geracao
fmean(1) = mean(fPop); [fbest(1),~] = max(fPop);
% plota populacao inicial
imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);

for it=1:max_it,

    % aplica o operador de mutação
    mPop = mutacao_rand_1(Pop,F,range);
    %mPop = mutacao_best_1(Pop,fPop,F);
    
    % aplica operador de crossover
    U = crossover_bin(Pop,mPop,CR); 
    
    % avalia a populacao
    fU = fitness(U);
    
    % aplica o operador de selecao gulosa (greedy selection)
    [Pop,fPop] = selecao_gulosa(Pop,fPop,U,fU);
    
    % encontre e salve o melhor fitness e o fitness medio desta geracao
    fmean(it+1) = mean(fPop); [fbest(it+1),idx] = max(fPop);
    
    % mostre o progresso
    imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);
    fprintf('Melhor indiv.: %3.5f %3.5f Melhor fitness: %f it: %d \n',Pop(idx,1),Pop(idx,2),fbest(it),it);
end
% salve os resultados principais na estrutura de saida
RESULT.Pop = Pop; RESULT.fPop = fPop; RESULT.fmean = fmean; RESULT.fbest = fbest;
