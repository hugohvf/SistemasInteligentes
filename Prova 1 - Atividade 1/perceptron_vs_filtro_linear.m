% Perceptron vs filtro linear utilizando LMS
% Problema de equalizacao de canal

clc;
clear all;
close all;

M = 3;  % dimensao de dados de entrada (portanto do filtro tambem)
N = 1e3;   % num. de sinais transmitidos (num. de padroes) no aprendizado
mu = 0.01;

sigma2_v = 0.05;   % variancia do ruido
h_canal = [1 ; 0.4];    % h = 1 + 0.4*z-1

% geracao dos dados
sn_N = sign(randn(N,1)); % randsrc(N,1);
vn_N = sqrt(sigma2_v)*randn(N,1); % normrnd(0,sqrt(sigma2_v),N,1);
sch_N = filter(h_canal,1,sn_N);
xn_N = sch_N + vn_N;

% organiza os dados na matriz Xn [M]:[N]
Xn = cria_Xn_equalizacao(xn_N,N,M);
Xn_p = [Xn ; ones(1,N)];  % ones eh o bias (que nao tem ruido)

% pre-alocacao de memoria e inicializacao
e2_filtro = zeros(N,1);
e2_perceptron = zeros(N,1);
Wn_filtro = zeros(M,N+1);
Wn_perceptron = zeros(M+1,N+1);
Yn_filtro = zeros(N,1);
Yn_perceptron = zeros(N,1);

%% treinamento do perceptron e do neuronio linear
for n=1:N,
    dn = sn_N(n,1);
    xn = Xn(:,n);
    xn_p = Xn_p(:,n);
    
    % filtro linear
    wn = Wn_filtro(:,n);
    yn = wn'*xn;
    erro = dn - yn;
    Wn_filtro(:,n+1) = wn + mu*erro*xn;
    e2_filtro(n,1) = erro^2;
    Yn_filtro(n,1) = yn;
    
    % perceptron
    wn = Wn_perceptron(:,n);
    yn = sign(wn'*xn_p);
    erro = dn - yn;
    Wn_perceptron(:,n+1) = wn + mu*erro*xn_p;
    e2_perceptron(n,1) = erro^2;
    Yn_perceptron(n,1) = yn;
end

%%
vetor_n = 1:N;

figure;
plot(vetor_n,e2_filtro,'b',vetor_n,e2_perceptron,'r')
title('curvas de erro quadrático');
legend('neuronio linear','perceptron')

figure;
plot(vetor_n,sch_N,'o');
xlabel('n'); ylabel('channel output');
title('channel output (without noise)');

figure;
plot(vetor_n,xn_N,'.');
xlabel('n'); ylabel('channel output');
title('channel output with noise)');

% figure;
% plot(real(xn_N),imag(xn_N),'.');
% title('channel output with noise')


figure;
plot(vetor_n,Yn_filtro','.',vetor_n,Yn_perceptron','.');
xlabel('n'); ylabel('equalizer output');
title('Equalizer output');
legend('Linear filter','Perpeptron')
axis([vetor_n(1) vetor_n(end) -2 2])

figure; hold on;
for ii=1:M,
    plot(vetor_n,Wn_filtro(ii,1:end-1));
end
xlabel('n'); ylabel('pesos');
title('Evolucao dos pesos do filtro');

hold off;

figure; hold on;
for ii=1:M+1,
    plot(vetor_n,Wn_perceptron(ii,1:end-1));
end
xlabel('n'); ylabel('pesos');
title('Evolucao dos pesos do perceptron');
hold off;

pause;

%% Teste
wn_f = Wn_filtro(:,end);
wn_p = Wn_perceptron(:,end);

N_block = 1e3;
blocks = 1e3;

decisaob_f = zeros(N_block,1);
decisaob_p = zeros(N_block,1);
erros_dec_f = 0;
erros_dec_p = 0;

%% uso do perceptron e do neuronio linear treinados
for b=1:blocks,
    % geracao dos dados do bloco
    sn_N = sign(randn(N_block,1)); % randsrc(N_block,1);
    vn_N = sqrt(sigma2_v)*randn(N_block,1); % normrnd(0,sqrt(sigma2_v),N_block,1);
    sch_N = filter(h_canal,1,sn_N);
    xn_N = sch_N + vn_N;
    
    % organiza os dados na matriz Xn [N]:[M]
    Xn = cria_Xn_equalizacao(xn_N,N_block,M);
    Xn_p = [Xn ; ones(1,N_block)];
    
    for n=1:N_block,
        dn = sn_N(n,1);
        xn = Xn(:,n);
        xn_p = Xn_p(:,n);
        
        % filtro linear
        decisaob_f(n,1) = sign(wn_f'*xn);
        
        % perceptron
        decisaob_p(n,1) = sign(wn_p'*xn_p);
    end
    erros_dec_f = erros_dec_f + sum(not(decisaob_f==sn_N));
    erros_dec_p = erros_dec_p + sum(not(decisaob_p==sn_N));
    if not(mod(b,100))
        fprintf('Bloco %d\n',b);
    end
end

Ber_f = erros_dec_f/(N_block*blocks);
Ber_p = erros_dec_p/(N_block*blocks);

fprintf('\nBER filtro     = %.4e\n',Ber_f);
fprintf('BER perceptron = %.4e\n',Ber_p);
fprintf('Num. de simbolos transmitidos = %d\n',N_block*blocks);
