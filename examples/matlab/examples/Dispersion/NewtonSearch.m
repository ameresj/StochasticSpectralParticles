function [ omegas ] = NewtonSearch( fun, n,maxit ,z0, r, tol )
%LuckStevensSearch Summary of this function goes here
%   Detailed explanation goes here


omegas=[]; %roots


% Determine number of zeroes
zk=exp(1i*2*sym('pi')*(0:n-1)'/n);
contour_integrand=(diff(fun,'omega')/fun);
contour_integral=vpa(r*mean(contour_integrand(r*zk+z0)));
contour_integral=contour_integral./(1j)
%numz=real(contour_integral);
numz=(real(contour_integral)+imag(contour_integral));

if (numz<0.5)
    omegas=[];
    return;
end


if numz<=1.5
    
    w=z0;
    newton_step=fun/diff(fun,'omega');
    for it=1:maxit
        
        w=w-vpa(newton_step(w));
        
        %if(abs(w-z0)>r)
        %   return; 
        %end
        
        res=abs(vpa(fun(w)));
        if (res<tol)
            omegas=w;
             plot(real(w),imag(w),'*'); hold on; drawnow;
            return;
        end
    end
    
    if (res>tol)
       fprintf('Maximum Iterations exceeded.\n') 
    end
    
return;  
end


z_new=z0+[(-1+1i),(1+1i),(1-1i),(-1-1i)]*(r/2)/sqrt(2);

for idx=1:4
    [omegas_new]=NewtonSearch(fun,n,maxit, z_new(idx),r/2,tol);
    omegas=[omegas, omegas_new];
end



end

