function [ omegas ] = ContourSearch( fun, n ,z0, r , tol )
%ContourSearch Summary of this function goes here
%   Detailed explanation goes here


omegas=[]; %roots


% Determine number of zeroes
zk=exp(1i*2*sym('pi')*(0:n-1)'/n);
contour_integrand=(diff(fun,'omega')/fun);
contour_integral=vpa(r*mean(contour_integrand(r*zk+z0)));
contour_integral=contour_integral./(1j);
numz=real(contour_integral);
% numz=(real(contour_integral)+imag(contour_integral));

if (numz<=tol)
    omegas=[];
    return;
end


if abs(numz-1)<=tol
    omegas=z0;
    plot(real(omegas),imag(omegas),'*'); hold on; drawnow;
    
    return;
end


z_new=z0+[(-1+1i),(1+1i),(1-1i),(-1-1i)]*(r/2)/sqrt(2);

for idx=1:4
    [omegas_new]=ContourSearch(fun,n, z_new(idx),r/2,tol);
    omegas=[omegas, omegas_new];
end



end

