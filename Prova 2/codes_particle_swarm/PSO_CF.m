% Particle swarm com fator de constricao
%Input: N - numero de particulas
%       dim - dimensao do espaco de busca
%       range - [x1min x1max; x2min x2max; ... ] [dim]x[2]
%       max_it - maximo num. de gerações/iteracoes
%       vmax - % maxima velocidade inicial do vetor velocidade
%       csi - fator de constricao
%       PLOT - variaveis de plotagem
%       .x - eixo x
%       .y - eixo y
%       .z - valor da funcao
%Output: RESULT - estrutura contendo resultados de interesse do PSO
%        .Pop - populacao final
%        .fPop - vetor de fitness final
%        .Mem - melhor posicao ja visitada por cada particula
%        .fMem - melhor fitness que cada particula ja teve
%        .fbest - evolucao do melhor fitness ([1]x[max_it]
%        .fmean - evolucao do fitness medio ([1]x[max_it]
%OBS: eh usada topologia com vizinhança em anel

function RESULT = PSO_CF(N,dim,range,max_it,vmax,csi,PLOT)

% gera populacao inicial
[Pop,fPop] = pop_inicial(N,range,dim);
% pre-alocacao de memoria para armazenamento de melhor fitness e medio
% medio em cada iteracao
fbest = zeros(1,max_it+1); fmean = zeros(1,max_it+1);
% encontre e salve o melhor fitness e o fitness medio desta geracao
fmean(1) = mean(fPop); [fbest(1),~] = max(fPop);

% topologia em anel: cria matriz com os indices dos vizinhos
vector = (1:N)';
Index = [circshift(vector,-1) circshift(vector,1)];
% encontrar melhor vizinho de cada celula 
[Bn,fBn] = best_initial(Pop,fPop,Index); 
% inicializar matriz de memoria da melhor posicao de cada particula
Mem = Pop; fMem = fPop;
% vetores de velocidade iniciais
V = 2*vmax*rand(N,dim) - vmax;
% plota populacao inicial
imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);

for it=1:max_it,
    % atualiza o vetor de velocidade de cada particula
    V = particle_direction(V,Mem,Bn,Pop,csi);
    % atualiza a posicao de cada particula na direcao V
    [Pop,fPop] = particle_move(Pop,V,range);
    % atualiza matriz de memoria da melhor posicao de cada particula
    II = fPop > fMem; Mem(II,:) = Pop(II,:); fMem(II) = fPop(II);
    % atualiza melhor vizinho de cada celula
    [Bn,fBn] = best_neighbor(Pop,fPop,Bn,fBn,Index);
    % encontre e salve o melhor fitness e o fitness medio desta geracao
    fmean(it+1) = mean(fPop); [fbest(it+1),idx] = max(fPop);
    % mostre o progresso
    imprime(PLOT.x,PLOT.y,PLOT.z,Pop(:,1),Pop(:,2),log(fPop),1,1);
    fprintf('Melhor indiv.: %3.5f %3.5f Melhor fitness: %f it: %d \n',Pop(idx,1),Pop(idx,2),fbest(it),it);
     
end
% salve os resultados principais na estrutura de saida
RESULT.Pop = Pop; RESULT.fPop = fPop;
RESULT.Mem = Mem; RESULT.fMem = fMem;
RESULT.fmean = fmean; RESULT.fbest = fbest;

