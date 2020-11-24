% script de visualizacao dos dados
close all; clear all; clc;

load data05.mat;

data = data05;
data(end,end) = 7;  % correcao de erro na base

% indice para selecionar atividades (7 ao total)
ind1=data(:,5)==1;
ind2=data(:,5)==2;
ind3=data(:,5)==3;
ind4=data(:,5)==4;
ind5=data(:,5)==5;
ind6=data(:,5)==6;
ind7=data(:,5)==7;

%elimina_redundancias;

figure;
plot(data(:,5));
title('Distribuicao das atividades')

% plota Ax,Ay,Az para cada atividade
% Atividade 1
figure;
subplot(3,1,1); plot(data(ind1,1),data(ind1,2));    % Ax
title(sprintf('Num. amostras(A%d): %d',1,sum(ind1)));    
subplot(3,1,2); plot(data(ind1,1),data(ind1,3));    % Ay
subplot(3,1,3); plot(data(ind1,1),data(ind1,4));    % Az

% Atividade 2
figure;
subplot(3,1,1); plot(data(ind2,1),data(ind2,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',2,sum(ind2)));
subplot(3,1,2); plot(data(ind2,1),data(ind2,3));    % Ay
subplot(3,1,3); plot(data(ind2,1),data(ind2,4));    % Az

% Atividade 3
figure;
subplot(3,1,1); plot(data(ind3,1),data(ind3,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',3,sum(ind3)));
subplot(3,1,2); plot(data(ind3,1),data(ind3,3));    % Ay
subplot(3,1,3); plot(data(ind3,1),data(ind3,4));    % Az

% Atividade 4
figure;
subplot(3,1,1); plot(data(ind4,1),data(ind4,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',4,sum(ind4)));
subplot(3,1,2); plot(data(ind4,1),data(ind4,3));    % Ay
subplot(3,1,3); plot(data(ind4,1),data(ind4,4));    % Az

% Atividade 5
figure;
subplot(3,1,1); plot(data(ind5,1),data(ind5,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',5,sum(ind5)));
subplot(3,1,2); plot(data(ind5,1),data(ind5,3));    % Ay
subplot(3,1,3); plot(data(ind5,1),data(ind5,4));    % Az

% Atividade 6
figure;
subplot(3,1,1); plot(data(ind6,1),data(ind6,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',6,sum(ind6)));
subplot(3,1,2); plot(data(ind6,1),data(ind6,3));    % Ay
subplot(3,1,3); plot(data(ind6,1),data(ind6,4));    % Az

% Atividade 7
figure;
subplot(3,1,1); plot(data(ind7,1),data(ind7,2));    % Ax
title(sprintf('Num. amostras (A%d): %d',7,sum(ind7)));
subplot(3,1,2); plot(data(ind7,1),data(ind7,3));    % Ay
subplot(3,1,3); plot(data(ind7,1),data(ind7,4));    % Az

%% Filtragem passa baixas
fs = 52;    % frequencia de amostragem Hz
fc = 3;     % frequencia de corte
Nf = 4;     % ordem do filtro butterworth
omega_c = 2*pi*fc;
% Projeto do filtro passa-baixas discreto Butterworth
B_butter = omega_c^(Nf);
poles = omega_c*exp(1i*(2*(0:Nf-1)+1)*pi/(2*Nf)+1i*pi/2);
A_butter = poly(poles);
T = 1/fs;
Omega_c = omega_c*T;
[Bd,Ad] = transf_bilinear(B_butter,A_butter,2/omega_c*tan(Omega_c/2));  % transformacao bilinear
% filtragem dos dados
dataf = data;
for kk=1:3
    dataf(:,kk+1) = real(filter(Bd,Ad,data(:,kk+1)));
end

% plota dados filtrados
for aa=1:7
    indx = strcat('ind',num2str(aa));
    figure;
    subplot(3,1,1); plot(data(eval(indx),1),dataf(eval(indx),2));    % Ax
    title(sprintf('Num. amostras (A%d): %d',aa,sum(eval(indx))));
    subplot(3,1,2); plot(data(eval(indx),1),dataf(eval(indx),3));    % Ay
    subplot(3,1,3); plot(data(eval(indx),1),dataf(eval(indx),4));    % Az
end

% breve comparacao do papel da filtragem
ind = ind5;
figure;
subplot(2,1,1); plot(data(ind,1),dataf(ind,2));    % Ax filtrado
title('Dado filtrado')
subplot(2,1,2); plot(data(ind,1),data(ind,2));    % Ax
title('Dado cru')