function [ omegas ] = Cz2Search( fun, n1, n2, z0, r, r_min,tol )
%Cz2Search Summary of this function goes here
%   Detailed explanation goes here




omegas=[]; %roots




if (r>r_min)
    z_new=z0+[(-1+1i),(1+1i),(1-1i),(-1-1i)]*(r/2)/sqrt(2);    
    for idx=1:4
        [omegas_new]=Cz2Search(fun,n1,n2, z_new(idx),r/2,r_min,tol);
        omegas=[omegas; omegas_new];
    end
    return;
end

% Determine number of zeroes
zk=exp(1i*2*sym('pi')*(0:n1-1)'/n1);

contour_integrand=(diff(fun,'omega')/fun);
contour_integral=vpa(r*mean(contour_integrand(r*zk+z0)));
contour_integral=contour_integral./(1j);


numz=(real(contour_integral)+imag(contour_integral));


if (numz<0.5)
    omegas=[];
    return;
end

K=min(ceil(numz),floor(n2/2)-1);




zk=exp(1i*2*sym('pi')*(0:n2-1)'/n2);

s = ifft(double(1./fun( zk*r+z0)));
H = hankel(s(2:K+1), s(K+1:2*K));
H2 = hankel(s(3:K+2), s(K+2:2*K+1));
omegas = eig(H2,H);

res=abs(vpa(fun(omegas)));
        
        
omegas=omegas(res<tol);
 plot(real(omegas),imag(omegas),'*'); hold on; drawnow;
if(~isempty(omegas))
 omegas=reshape(omegas,numel(omegas),1);
end
end

