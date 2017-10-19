function [ omegas ] = McCuneSearch( fun, n1,n2 ,z0, r, tol )
%LuckStevensSearch Summary of this function goes here
%   Detailed explanation goes here


omegas=[]; %roots
% 
% theta=linspace(0,2*pi);
% plot(real(z0)+r*cos(theta),imag(z0)+r*sin(theta)); hold on; drawnow;


% Determine number of zeroes
zk=exp(1i*2*sym('pi')*(0:n1-1)'/n1);
contour_integrand=(diff(fun,'omega')/fun);
contour_integral=double(r*mean(contour_integrand(r*zk+z0)))./1j;



fprintf('%g \n',abs(contour_integral));
% contour_integral

numz=real(contour_integral);%+imag(contour_integral);

if (numz<=0.5)
    omegas=[];
    return;
end


if numz<=1.5
    
    zk=exp(1i*2*sym('pi')*(0:n2-1)'/n2);
    fk=double(fun(r*zk+z0));
    
    c = fft(fk)/n2;
    cp = (1:n2-1)'.*c(2:end);
    ppzk = n2*ifft([cp;0]);
    w = mean(zk.^2.*ppzk./fk)*r + z0;
    res=double(fun(w));
    
%     if (res<tol)
        omegas=w;
         return;
%    end
end

z_new=z0+[(-1+1i),(1+1i),(1-1i),(-1-1i)]*(r/2)/sqrt(2);

for idx=1:4
    [omegas_new]=McCuneSearch(fun,n1,n2, z_new(idx),r/2,tol);
    omegas=[omegas, omegas_new];
end


 plot(real(omegas),imag(omegas),'*'); hold on; drawnow;

end

