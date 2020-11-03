function Xn = cria_Xn_equalizacao(xn_N,N,M)
% cria matriz de sinais recebidos dada a ordem M
% do filtro (canal) e o num. de amostras transmitidas N
% N = num. de sinais transmitidos no aprendizado
% M = ordem do filtro (canal)

xn_N = xn_N(:); % garante que xn_N eh vetor coluna

% alocacao de memoria
Xn = zeros(M,N);
for n=1:N,
    % transiente de preenchimento dos taps (instantes) do filtro
    if n < M
        Xn(1:n,n) = xn_N(n:-1:1,1); % passado para baixo [x(n);x(n-1);...;x(n-(M-1))]
    end
    
    % regime de funcionamento dos taps do filtro
    if n >= M
        Xn(:,n) = xn_N(n:-1:n-M+1,1);
    end
end

end