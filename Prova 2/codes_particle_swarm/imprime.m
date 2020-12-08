% Plota a população e seu fitness na superfície
function [] = imprime(vx,vy,vz,x,y,fx,it,mit)

if rem(it,mit) == 0,
    temporizador(0.1);  % espera x segundos
    mesh(vx,vy,vz); hold on;
    xlabel('x'); ylabel('y'); zlabel('f(x,y)');
    plot3(x,y,fx,'k*'); drawnow; hold off;
end

end
function [] = temporizador(tempo)
    tic;
    while toc < tempo,
    end
end
