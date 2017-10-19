function [ roots ] = LuckStevensSearch( fun, n, z0, r )
%LuckStevensSearch Summary of this function goes here
%   Detailed explanation goes here


roots=[];

theta=linspace(0,2*pi);
plot(real(z0)+r*cos(theta),imag(z0)+r*sin(theta)); hold on; drawnow;

zk=exp(1i*2*sym('pi')*(0:n-1)/n);


%Estimate the numbers of zeroes in the box with 
contour_integrand=(diff(fun,'omega')/fun/(1j));
contour_integral=double(r*mean(contour_integrand(r*zk+z0)));
%contour_integral=abs(contour_integral);

fprintf('%g \n',abs(contour_integral));

if (abs(contour_integral)<1/n)
    root_new=[];
elseif abs(contour_integral)<=1 
    % LuckStevens
    root_new=double(mean((zk).^2./fun(r*zk+z0))./mean(zk./fun(r*zk+z0)));
    roots=[roots, root_new];
else  
   z_new=z0+[(-1+1i),(1+1i),(1-1i),(-1-1i)]*(r/2)/sqrt(2);

   
   for idx=1:4
      [root_new]=LuckStevensSearch(fun,n, z_new(idx),r/2);
      roots=[roots, root_new];
   end
end


end

