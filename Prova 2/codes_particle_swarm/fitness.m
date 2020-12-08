% funcao de fitness - multimodal com duas dimensoes
%Input: x - populacao [N]x[dim]
%Output: fx - valor de fitness

function fx = fitness(x)
 
% funcao
fx = exp(x(:,1).*sin(4*pi*x(:,1))...
    - x(:,2).*sin(4*pi*x(:,2) + pi) + 1);
