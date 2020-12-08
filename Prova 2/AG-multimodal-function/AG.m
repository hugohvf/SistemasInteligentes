%J. Holland, "Adaptation in Natural and Artificial Systems", 1975
%Input: N - numero de individuos
%       dim - dimensao do espaço de busca
%       range - [x1min x1max; x2min x2max; ... ] [dim]x[2]
%       pc - probabilidade de crossover
%       pm - probabilidade de mutação
%       max_it - máximo num. de gerações/iterações
%       q - num. de individuos em cada torneio
%       PLOT - variaveis de plotagem
%       .x - eixo x
%       .y - eixo y
%       .z - valor da função
%Output: RESULT - estrutura contendo resultados de interesse do GA
%        .Pop - população final
%        .fPop - vetor de fitness final
%        .fbest - evolução do melhor fitness ([1]x[max_it]
%        .fmean - evolução do fitness médio ([1]x[max_it]

function RESULT = AG(Np,dim,range,pc,pm,max_it,q,PLOT)

% gera populacao inicial
[Pop,fPop] = pop_inicial(Np,range,dim);
% pre-alocacao de memoria para armazenamento de melhor fitness e fitness
% medio em cada iteracao
fbest = zeros(1,max_it); fmean = zeros(1,max_it);
% plota populacao inicial
imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);

for it=1:max_it,
    
    % abordagem elitista - preserva uma copia do melhor individuo
    [vbest,id] = max(fPop); best = Pop(id,:);

    % aplica o operador de recombinacao
    Temp = crossover(Pop,pc);
    
    % aplica operador de mutacao gaussiana
    Temp = mutacao_gaussiana(Temp,pm,range); 
    
    % avalia a populacao
    fTemp = fitness(Temp);
    
    % aplica o operador de selecao por torneio
    [Pop,fPop] = torneio(Temp,fTemp,q,Np);
    % elitismo - encontre o pior individuo e substitua-o pelo melhor ate o
    % momento
    [~,id] = min(fPop); Pop(id,:) = best; fPop(id) = vbest;
    
    % encontre e salve o melhor fitness e o fitness medio desta geracao
    fmean(it) = mean(fPop); [fbest(it),idx] = max(fPop);
    
    % mostre o progresso
    imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);
    fprintf('Melhor indiv.: %3.5f %3.5f Melhor fitness: %f it: %d \n',Pop(idx,1),Pop(idx,2),fbest(it),it);
end
% salve os resultados principais na estrutura de saida
RESULT.Pop = Pop; RESULT.fPop = fPop; RESULT.fmean = fmean; RESULT.fbest = fbest;
