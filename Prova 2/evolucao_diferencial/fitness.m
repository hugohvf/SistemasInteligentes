% fun��o de fitness - multimodal com duas dimensoes
%Input: x - popula��o [N]x[dim]
%Output: fx - valor de fitness

function fx = fitness(x)
 
% fun��o
fx = exp(x(:,1).*sin(4*pi*x(:,1))...
    - x(:,2).*sin(4*pi*x(:,2) + pi) + 1);
