% Atualiza o vetor de velocidade de cada particula
%Input: V - vetores de velocidade atuais
%       Mem - melhor posicao ja visitada por cada particula
%       Bn - melhor vizinho de cada particula
%       Pop - populacao de particulas
%       csi - fator de constricao
%Output: V - vetores de velocidade atualizados

function V = particle_direction(V,Mem,Bn,Pop,csi)

% constantes de aceleracao (soma = 4.1)
L1 = 2.05; L2 = 2.05;
% avalia dimensoes
[N,dim] = size(Pop);
% atualiza o vetor de velocidade de cada particula
V = csi*(V + L1*rand(N,dim).*(Mem - Pop) + L2*rand(N,dim).*(Bn - Pop));