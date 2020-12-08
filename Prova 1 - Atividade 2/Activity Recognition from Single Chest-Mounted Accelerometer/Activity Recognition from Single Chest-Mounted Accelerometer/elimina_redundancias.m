% elimina redundancias
datatmp = [];
for kk=1:7
    indx = strcat('ind',num2str(kk));
    temp = data(eval(indx),:);
    N = size(temp,1);
    temp1 = temp;
    anterior = 0; cont = 0;
    for ii=1:N,
        if anterior < ii
            cont = cont + 1;
            temp1(cont,:) = temp(ii,:);
        end
        anterior = temp(ii,1);
    end
    N = cont;
    datatmp = [datatmp ; temp1];
end