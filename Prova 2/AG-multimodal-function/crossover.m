% Operador de crossover: define os pares e executa a recombinação
%Input: Pop - população de individuos
%       pc - probabilidade de um individuo realizar a reprodução

function Temp = crossover(Pop,pc)

% verifica o numero de individuos
[Np,dim] = size(Pop);

% gera aleatoriamente os pares de individuos para o crossover (com reposicao)
C = zeros(Np,2);
C(:,1) = rand(Np,1); C(:,2) = round((Np-1)*rand(Np,1))+1;
% matriz Ip indica os pares de individuos a serem recombinados
I = find(C(:,1) <= pc); Ip = [I C(I,2)];
% matriz Temp acumula os pais e os filhos
Temp = [Pop; zeros(2*Np,dim)]; cont = 0;
if ~isempty(Ip),
    for ii=1:length(Ip(:,1)),   %1:2:length(Ip(:,1)),
        % testa se os individuos sao distintos
        if Ip(ii,1)==Ip(ii,2)   
            continue;
        end
        % aplica o crossover aritmetico
        [f1,f2] = crossover_aritmetico(Pop(Ip(ii,1),:), Pop(Ip(ii,2),:));
        % insere os filhos na populacao (aumenta a populacao)
        %Temp = [Temp;f1;f2];
        Temp(Np+2*cont+1:Np+2*cont+2,:) = [f1;f2];
        cont = cont + 1;
    end
end
cont = cont - 1;
Temp = Temp(1:Np+2*cont+2,:);
end