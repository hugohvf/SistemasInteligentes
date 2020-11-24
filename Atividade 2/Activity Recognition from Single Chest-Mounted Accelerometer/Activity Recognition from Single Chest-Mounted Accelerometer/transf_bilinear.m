function [Bd, Ad] = transf_bilinear(B,A,T)
% Transformacao bilinear do dominio s (continuo) para o dominio z
% (discreto)
% Entradas:     B = vetor dos coeficientes do numerador em s
%               A = vetor dos coeficientes do denominador em s
%               T = periodo de amostragem
% Saidas:       Bd = vetor dos coeficientes do numerador em z
%               Ad = vetor dos coeficientes do denominador em z

N = length(A);
M = length(B);

if (M>N)
    error('Ordem do numerador nao pode exceder a ordem do denominador');
end

z = roots(B);   % raizes do numerador (raizes de H(s))
p = roots(A);   % raizes do denoninador (polos de H(s))
gain = real(B(1)/A(1)*prod(2/T-z)/prod(2/T-p));

% raizes mapeadas no dominio discreto z
zd = (1+z*T/2)./(1-z*T/2);
pd = (1+p*T/2)./(1-p*T/2);

Bd = gain*poly([zd;-ones(N-M,1)]);
Ad = poly(pd);