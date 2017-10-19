%% Particle in Fourier 1D2V for Vlasov-Maxwell
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%% Features
%
% * GEMPIC Hamiltionian Splitting (https://arxiv.org/abs/1609.03053)
% * Partile in Fourier field discretization
% * Fourth order composition of second order Strang splitting
%
%% Equations
% This code solves the three dimensional 1D2V Vlasov-Maxwell system
% using Lagrangian particles for the Vlasov equation and
% a spectral discretization of Maxwells equations.
% The Vlasov equation reads.
%
% $$\partial_t f + v_1 \partial_x f + \frac{q_s}{m_s}
%  \left(  E_1 \partial_{v_1} f  +  E_2 \partial_{v_2} f   \right)
% + \frac{q_s}{m_s} B \left(  v_2 \partial_{v_1}
%   f - v_1 \partial_{v_2} f   \right) =0$$
%
% It is coupled to Maxwells equations
%
% $$ \begin{array}{cl}
% \partial_t B &= - \partial_x E_2 ,\\
% \partial_t E_1 &= -j_1,\\
% \partial_t E_2 &= - \partial_x B - j_2,\\
% \partial_x E_1 &= \rho
% \end{array} $$
%
% The currents $j_1(x,t),j_2(x,t)$ and the charge density
% $\rho(x,t)$ are defined as.
%
% $$ \begin{array}{cl}
%  j_1(x,t) &:=  \int v_1 f(x,v_1,v_2,t) \mathrm{d}v_1 \mathrm{d}v_2,\\
%  j_2(x,t) &:=  \int v_2 f(x,v_1,v_2,t) \mathrm{d}v_1 \mathrm{d}v_2,\\
%  \rho(x,t) &:= \int f(x,v_1,v_2,t) \mathrm{d}v_1 \mathrm{d}v_2
%  \end{array} $$
%
% The general initial condition setting parameters for all test cases are.
%
% $$ f(x,v,t=0)=  \frac{1+ \epsilon \cos(k x)}{2\pi \sigma_1 \sigma_2^2 }
%  \mathrm{e}^{ - \frac{v_1^2}{2\sigma_1^2}  }
% \left(
% \delta \mathrm{e}^{ - \frac{(v_2 -v_{0,1})^2}{2\sigma_2^2}  }
% +(1-\delta)\mathrm{e}^{ - \frac{(v_2 -v_{0,2})^2}{2\sigma_2^2}}
% \right) $$
%
% $$ \begin{array}{cl}
% \partial_x E_1(x,0) &= \rho(x,0),\\
%  E_2(x,0) &= 0,\\
%  B(x,0) &= \beta_r \cos(kx) +\beta_i \sin(kx)
% \end{array} $$
%

clear all; close  all;
%% Solver parameters
Np=1e4;    % Number of Lagrangian particles
Nf=1;      % Number of Fourier Modes
dt=0.05;   % Time step


q=-1;   % charge
m=1;    % mass
mu0=1;   %Vacuum permeability
%%
% Available test cases are:
%
% * landau - strong Landau damping
% * weibel - the Weibel instability
% * weibels - the streaming Weibel instability
%
testcase='weibel'; % 'weibel','landau','weibels'


switch (testcase)
    
    case('weibel')
        % Weibel instability
        eps=0; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
        betar=sign(q)*1e-3; betai=0;
        k=1.25;    % Wave vector
        sigma1=0.02/sqrt(2);
        sigma2=sqrt(12)*sigma1;
        sigma3=sigma1;
        v01=0;
        v02=0;
        delta=0;
        tmax=200;
                k1=k;k2=k;k3=k;

    case ('weibels')
        % Streaming Weibel instability
        sigma1=0.1/sqrt(2);
        sigma2=sigma1; sigma3=sigma1;
        k=0.2;
        betai=sign(q)*1e-3; betar=0;
        v01=0.5;
        v02=-0.1;
        delta=1/6.0;
        eps=0;
        tmax=400;
        sigma3=sigma1;
        k1=k;k2=k;k3=k;
    case('landau')
        % Strong Landau damping
        eps=0.1;
        k=0.5;
        sigma1=1;sigma2=1; sigma3=1;
        betar=0;betai=0;v01=0;v02=0;delta=0;
        tmax=10;
        k1=k;k2=k;k3=k;
    case('alfven')
       q=1;
        eps=0;
        k=0.;
         sigma1=1;sigma2=1;
        betar=1e-2;
        betai=0;
        v01=0;v02=0;delta=0;
        tmax=20;
        sigma3=sigma1;
        k1=k;k2=k;k3=k;
end





% Length of domain
L1=2*pi/k1;  L2=2*pi/k2; L3=2*pi/k3;
% Initial condition
f0=@(x1,x2,x3,v1,v2,v3) (1+ eps*cos(k*x2)).*...
     exp(- (v1.^2./sigma1^2 + v3.^2/sigma3^2)/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))...
    *(2*pi).^(-1.5)/sigma3/sigma1/sigma2;




%% Initialize particles - Sampling
sbl=sobolset(6,'Skip',2);
xk1=sbl(1:Np,1)*L1; % Uniformly distributed U(0,L)
xk2=sbl(1:Np,2)*L2;
xk3=sbl(1:Np,3)*L3;


vk1=norminv(sbl(1:Np,4))*sigma1;  % Normally distributed $N(0,\sigma)$
vk2=norminv(sbl(1:Np,5))*sigma2;
vk3=norminv(sbl(1:Np,6))*sigma3;

vk2(1:floor(delta*Np))=vk2(1:floor(delta*Np)) + v01;
vk2(floor(delta*Np)+1:end)=vk2(floor(delta*Np)+1:end) +v02;

g0=@(x1,x2,x3,v1,v2,v3) exp(- (v1.^2./sigma1^2 + v3.^2/sigma3^2)/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))...
    *(2*pi).^(-1.5)/sigma3/sigma1/sigma2/L1/L2/L3;


h0=@(x1,x2,x3,v1,v2,v3) exp(- (v1.^2./sigma1^2 + v2.^2/sigma2^2 + ...
    v3.^2/sigma3^2)/2 )*(2*pi).^(-1.5)/sigma3/sigma1/sigma2;

% Charge and mass for every particle
qk=ones(Np,1)*q;
mk=ones(Np,1)*m;

fk=f0(xk1,xk2,xk3,vk1,vk2,vk3);       % Vlasov likelihood
gk=g0(xk1,xk2,xk3,vk1,vk2,vk3);       % Particle (Sampling) likelihood

%% Initialize Particle in Fourier Maxwell Solver
% Spatial wave vector
  kx1=(2*pi)/L1*(0:Nf)';
%  kx1=(2*pi)/L1*(-Nf:Nf)';
 kx2=(2*pi)/L2*([-Nf:Nf])';
% kx3=(2*pi)/L3*([-Nf:Nf])';
% kx2=0;
% kx1=0;
kx3=0;

[KX1,KX2,KX3]=ndgrid(kx1,kx2,kx3); KX1=KX1(:);KX2=KX2(:);KX3=KX3(:);
% Masks to filter out zeroth mode
KDX=~( KX1==0 & KX2==0 & KX3==0);
KDX1=( KX1~=0); KDX2=( KX2~=0); KDX3=( KX3~=0);

% Particle weight
wk=fk./gk;
% 
% vk1=vk1-mean(wk.*vk1)/Np;
% vk2=vk2-mean(wk.*vk2)/Np;
% vk3=vk3-mean(wk.*vk3)/Np;
% repmat(kx1, [1, length(kx2)])


% Fourier Coefficients of the Fields
E1=zeros(length(KX1),1); E2=E1; E3=E1;
B1=zeros(length(KX1),1); B2=B1; B3=B1;
PHI=E1;
% background current
J0BG=zeros(3,1)*L1*L2*L3;

%%
% Splitting constants for composition methods. Set to scalar 1
% if only Strang splitting is desired
comp_gamma=[1/(2-2^(1/3)), -2^(1/3)/(2-2^(1/3)),  1/(2-2^(1/3))];
comp_gamma=1;

%% Initializing Maxwell Solver
% Initial condition on k-th Fourier mode of B3
B3( KX1==k & KX2==0 & KX3==0  )=betar + 1j*betai;
B3(KX1==-k & KX2==0 & KX3==0 )=betar - 1j*betai;

% Solve poisson equation for intialization
rhok=mean(bsxfun(@times,qk.*wk, exp(-1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ) ) )).'/L1/L2/L3;

PHI(KDX)=1./(KX1(KDX).^2+KX2(KDX).^2+KX3(KDX).^2).*rhok(KDX); 
E1=-PHI.*1j.*KX1; E2=-PHI.*1j.*KX2; E3=-PHI.*1j.*KX3;
% E1(KDX1)=1./(1j*KX1(KDX1)).*rhok(KDX1);
%  E2(KDX2)=1./(1j*KX2(KDX2)).*rhok(KDX2);
%  E3(KDX3)=1./(1j*KX3(KDX3)).*rhok(KDX3);


% Allocate diagnostic output
Nt=ceil(tmax/dt);
Epot=zeros(Nt,3);Ekin=Epot;Bpot=zeros(Nt,3);Momentum=Epot;

% Evaluate mode and it's complex conjugate at once
% evE=@(a,b) (real(a)*real(b) - imag(a)*imag(b)).*2.0;
% Evaluation with complex symmetry around x axis
evE=@(a,b) real( a(:,~KDX1)*b(~KDX1)  )+ ...
    (real( a(:,KDX1) )*real(b(KDX1)) - ...
    imag(a(:,KDX1))*imag(b(KDX1))).*2.0;
evE_mask=@(a,b,KMASK) real( a(:,~KDX1 & KMASK)*b(~KDX1 & KMASK)   )+ ...
    (real( a(:,KDX1 & KMASK) )*real(b(KDX1 & KMASK)) - ...
    imag(a(:,KDX1 & KMASK))*imag(b(KDX1 & KMASK))).*2.0;
% evE=@(a,b) real( a*b  );
% evE_mask=@(a,b,KMASK) real( a(:, KMASK)*b( KMASK)   );

% scalar product
scalP=@(A,B) real( A(~KDX1)'*B(~KDX1))...
          +(real( A(KDX1)' )*real(B(KDX1))-imag(A(KDX1)')*imag(B(KDX1)))*2;

% Determine scatter plot limits
vmax1=max(abs(vk1)); vmax2=max(abs(vk2));
runtime=tic;

psik0=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
psik1=psik0;
get_weight= @(fk,gk,xk1,xk2,xk3,vk1,vk2,vk3) fk./gk;


for tdx=1:Nt
    t=tdx*dt;
    % Composition steps of the second order Strang splitting
    for splitdx=1:length(comp_gamma)
        deltat=comp_gamma(splitdx)*dt;

        
        
        %H_B
        E1 = E1 + (deltat/2)*1j*( KX2.*B3 - KX3.*B2);
        E2 = E2 + (deltat/2)*1j*( KX3.*B1 - KX1.*B3);
        E3 = E3 + (deltat/2)*1j*( KX1.*B2 - KX2.*B1);
        
        %H_E
        vk1=vk1 + (deltat/2)*evE(psik0,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evE(psik0,E2).*qk./mk;
        vk3=vk3 + (deltat/2)*evE(psik0,E3).*qk./mk;
        
        B1 = B1 - (deltat/2)*1j*( KX2.*E3 - KX3.*E2);
        B2 = B2 - (deltat/2)*1j*( KX3.*E1 - KX1.*E3);
        B3 = B3 - (deltat/2)*1j*( KX1.*E2 - KX2.*E1);

        
        
        %H_p3
        xk3=xk3+ (deltat/2)*vk3;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk1=vk1+ (-qk./mk).*( evE_mask( psik1-psik0, B2./(1j*KX3), KDX3 )...
                  + deltat/2*vk3.*evE_mask(psik1, B2,~KDX3) );
        vk2=vk2+ (qk./mk).*(evE_mask( psik1-psik0, B1./(1j*KX3), KDX3 )...
                + deltat/2*vk3.*evE_mask(psik1, B1,~KDX3) );
            
        E3(KDX3)=E3(KDX3) + mean(bsxfun(@times, wk.*qk,...
                   conj(psik1(:,KDX3) -psik0(:,KDX3)) )).'./(1j.*KX3(KDX3))...
            /L1/L2/L3;
        E3(~KDX3)= E3(~KDX3) - (deltat/2)*mean(bsxfun(@times, ...
            wk.*vk3.*qk, conj(psik1(:,~KDX3)))).'/L1/L2/L3;
        E3(~KDX)=0;
        psik0=psik1;

        %H_p2
        xk2=xk2+ (deltat/2)*vk2;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk1=vk1+ (qk./mk).*(evE_mask( psik1-psik0, B3./(1j*KX2),KDX2 ) ...
             + deltat/2*vk2.*evE_mask(psik1, B3,~KDX2));
        vk3=vk3+ (-qk./mk).*(evE_mask( psik1-psik0, B1./(1j*KX2),KDX2 )...
            + deltat/2*vk2.*evE_mask(psik1, B1,~KDX2));
         
        E2(KDX2)=E2(KDX2) + mean(bsxfun(@times, wk.*qk,...
            conj(psik1(:,KDX2) -psik0(:,KDX2)) )).'./(1j.*KX2(KDX2))...
            /L1/L2/L3;
        E2(~KDX2)= E2(~KDX2) - (deltat/2)*mean(bsxfun(@times, ...
              wk.*vk2.*qk, conj(psik1(:,~KDX2)))).'/L1/L2/L3;
        E2(~KDX)=0;
        psik0=psik1;
        
        %H_p1
        xk1=xk1+ (deltat)*vk1;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk2=vk2 + (-qk./mk).*(evE_mask( psik1-psik0, B3./(1j*KX1),KDX1 ) ...
            + deltat*vk1.*evE_mask(psik1, B3,~KDX1));
        vk3=vk3+ (qk./mk).*(evE_mask( psik1-psik0, B2./(1j*KX1),KDX1 )...
            + deltat*vk1.*evE_mask(psik1, B2,~KDX1));
        
        E1(KDX1)=E1(KDX1) + mean(bsxfun(@times, wk.*qk,...
            conj(psik1(:,KDX1) -psik0(:,KDX1)) )).'./(1j.*KX1(KDX1))...
            /L1/L2/L3;
        E1(~KDX1)= E1(~KDX1) - (deltat)*mean(bsxfun(@times, ...
            wk.*vk1.*qk, conj(psik1(:,~KDX1)))).'/L1/L2/L3;
        E1(~KDX)=0;
        psik0=psik1;
        
        %H_p2
        xk2=xk2+ (deltat/2)*vk2;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk1=vk1+ (qk./mk).*(evE_mask( psik1-psik0, B3./(1j*KX2),KDX2 ) ...
            + deltat/2*vk2.*evE_mask(psik1, B3,~KDX2));
        vk3=vk3+ (-qk./mk).*(evE_mask( psik1-psik0, B1./(1j*KX2),KDX2 )...
            + deltat/2*vk2.*evE_mask(psik1, B1,~KDX2));
%          
        E2(KDX2)=E2(KDX2) + mean(bsxfun(@times, wk.*qk,...
            conj(psik1(:,KDX2) -psik0(:,KDX2)) )).'./(1j.*KX2(KDX2))...
            /L1/L2/L3;
        E2(~KDX2)= E2(~KDX2) - (deltat/2)*mean(bsxfun(@times, ...
            wk.*vk2.*qk, conj(psik1(:,~KDX2)))).'/L1/L2/L3;
         E2(~KDX)=0;
        psik0=psik1;
        
%         %H_p3
        xk3=xk3+ (deltat/2)*vk3;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk1=vk1+ (-qk./mk).*(evE_mask( psik1-psik0, B2./(1j*KX3), KDX3 )...
                + deltat/2*vk3.*evE_mask(psik1, B2,~KDX3));
        vk2=vk2+ (qk./mk).*(evE_mask( psik1-psik0, B1./(1j*KX3), KDX3 )...
                + deltat/2*vk3.*evE_mask(psik1, B1,~KDX3));
        E3(KDX3)=E3(KDX3) + mean(bsxfun(@times, wk.*qk,...
            conj(psik1(:,KDX3) -psik0(:,KDX3)) )).'./(1j.*KX3(KDX3))...
            /L1/L2/L3;
        E3(~KDX3)= E3(~KDX3) - (deltat/2)*mean(bsxfun(@times, ...
            wk.*vk3.*qk, conj(psik1(:,~KDX3)))).'/L1/L2/L3;
        E3(~KDX)=0;
        psik0=psik1;  
        
        
        
        %H_E
        vk1=vk1 + (deltat/2)*evE(psik0,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evE(psik0,E2).*qk./mk;
        vk3=vk3 + (deltat/2)*evE(psik0,E3).*qk./mk;
        
        B1 = B1 - (deltat/2)*1j*( KX2.*E3 - KX3.*E2);
        B2 = B2 - (deltat/2)*1j*( KX3.*E1 - KX1.*E3);
        B3 = B3 - (deltat/2)*1j*( KX1.*E2 - KX2.*E1);

        %H_B
        E1 = E1 + (deltat/2)*1j*( KX2.*B3 - KX3.*B2);
        E2 = E2 + (deltat/2)*1j*( KX3.*B1 - KX1.*B3);
        E3 = E3 + (deltat/2)*1j*( KX1.*B2 - KX2.*B1); 
        

    end
    
    xk1=mod(xk1,L1); xk2=mod(xk2,L2);xk3=mod(xk3,L3);
    %% Scatter plot of Lagrangian particles
        if (mod(t,tmax/10)==0)
%     scatter(xk1,vk1,8,fk,'filled');
%     colorbar;
%     drawnow;
    
%              scatter3(xk1,vk1,vk2,8,fk,'filled')
%             colorbar;
%             axis([0,L1,-vmax1,vmax1,-vmax2,vmax2])
%             xlabel('x');ylabel('v_1');zlabel('v_2');
%             title(sprintf('t=%g,',t))
%             drawnow; hold off;
    
    % [SX,SY,SZ]=sphere(14);
    % SC=ones(size(SX));
    % sz=0.1/max([L,vmax1*2,vmax2*2]);
    % szx=sz*L;
    % szy=sz*(vmax1*2);
    % szz=sz*(vmax2*2);
    % for idx=1:Np/100
    %
    %
    % surf(SX*szx+xk(idx),SY*szy+vk1(idx),SZ*szz+vk2(idx),SC.*fk(idx),...
    %     'LineStyle','none'); hold on;
    %
    % end
    % axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
    %      xlabel('x');ylabel('v_1');zlabel('v_2');
    % light('Position',[1 -1 1],'Style','infinite','Color',[1 1 1]);
    % lighting gouraud;
    % % view(30,30)
    %
    % hold off;
    
        end
    
    
    Epot(tdx,1) =(E1'*E1)*(L1*L2*L3*2)/2;
    Epot(tdx,2) =(E2'*E2)*(L1*L2*L3*2)/2;
    Epot(tdx,3) =(E3'*E3)*(L1*L2*L3*2)/2;
    
    Ekin(tdx,1) = mean(vk1.^2.*wk)/2;
    Ekin(tdx,2) = mean(vk2.^2.*wk)/2;
    Ekin(tdx,3) = mean(vk3.^2.*wk)/2;
    
    Momentum(tdx,1)=mean(vk1.*wk)+(scalP(E2,B3)-scalP(E3,B2))*L1*L2*L3;
    Momentum(tdx,2)=mean(vk2.*wk)+(scalP(E3,B1)-scalP(E1,B3))*L1*L2*L3;
    Momentum(tdx,3)=mean(vk3.*wk)+(scalP(E1,B2)-scalP(E2,B1))*L1*L2*L3;
              
    Bpot(tdx,1)=(B1'*B1)*(L1*L2*L3*2)/2;
    Bpot(tdx,2)=(B2'*B2)*(L1*L2*L3*2)/2;
    Bpot(tdx,3)=(B3'*B3)*(L1*L2*L3*2)/2;
end
runtime=toc(runtime);
%
%% Field Energys
time=0:dt:tmax-dt;
figure;
semilogy(time,Epot); hold on;
semilogy(time,Bpot)
xlabel('time'); grid on;
title('field energy');
l=legend('$\frac{1}{2} || E_1 ||^2$','$\frac{1}{2} || E_2 ||^2$',...
    '$\frac{1}{2} || E_3 ||^2$',...
    '$\frac{1}{2} || B_1 ||^2$',...
    '$\frac{1}{2} || B_2 ||^2$',...
    '$\frac{1}{2} || B_3 ||^2$',...
    'Location','SouthEast');
set(l,'Interpreter','LaTeX','FontSize',16)
%
%% Energy Error
figure
energy=sum(Epot,2)+sum(Bpot,2)+sum(Ekin,2);


semilogy(time, abs((energy-energy(1)))/energy(1)) ;hold on;
grid on;
axis tight;
title('relative energy error');
xlabel('time');
ylabel('$|\mathcal{H}(t)-\mathcal{H}(0)|/\mathcal{H}(0)$',...
    'interpreter','latex')
%
%% Momentum Error
figure;
semilogy(time, abs(Momentum-Momentum(1,:)))
legend('P_1','P_2'); grid on;
xlabel('time');
title('absolute momentum error');
axis tight
%
% %% Performance Statistics
% % For one stage of the integrator
% fprintf('Overall time: %g s\n',runtime);
% fprintf('Pushed %g particles/s \n', Np*Nt/runtime*2*length(comp_gamma));
%
%
%
%% Poisson equation
% The Poisson equation is conserved by the Hamiltonian splitting. This can
% be checked here. In order to check the vadility of the solution the
% variance on the field $E_1$ can be calculated.
%
% % Samples
% rhok=mean(bsxfun(@times,qk.*wk, exp(-1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ) ) )).'/L1/L2/L3;

% rhok
% E1.*(1j*KX1)+E2.*(1j*KX2)+E3.*(1j*KX3)
%
% E1(KDX1)=
% E2(KDX2)=-1./(KX1(KDX2).^2+KX2(KDX2).^2+KX3(KDX2).^2).*1j.*KX1(KDX2).*rhok(KDX2);
% E3(KDX3)=-1./(KX1(KDX3).^2+KX2(KDX3).^2+KX3(KDX3).^2).*1j.*KX1(KDX3).*rhok(KDX3);


% % Solve poisson equation for intialization
rhok=mean(bsxfun(@times,qk.*wk, exp(-1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ) ) )).'/L1/L2/L3;

% PHI(KDX)=1./(KX1(KDX).^2+KX2(KDX).^2+KX3(KDX).^2).*rhok(KDX);
% E1_=-PHI.*1j.*KX1;E2_=-PHI.*1j.*KX2;E3_=-PHI.*1j.*KX3;
% 
% abs(E1_-E1)
% abs(E2_-E2)
% abs(E3_-E3)

gauss_error=E1*1j.*KX1+E2*1j.*KX2+E3*1j.*KX3-rhok; 
gauss_error=norm(gauss_error(KDX))
%Divergence of B
norm(B1*1j.*KX1+B2*1j.*KX2+B3*1j.*KX3)

% yy=bsxfun(@times, bsxfun(@times,qk.*wk, exp(-1j*( xk1*KX1(KDX1).'+xk2*KX2(KDX2).'+xk3*KX3(KDX3).' ) ) ).'/L1/L2/L3,...
%      -1./(KX1(KDX1).^2+KX2(KDX2).^2+KX3(KDX1).^2).*1j.*KX1(KDX1)).';
%  yy_cov=cov(yy)/Np; % Sample covariance
%
% L2 error on the Fourier modes
% error_pos=norm(E1(KDX1).'-mean(yy)) ;
% % Linearized propagation of uncertainty
%  error_pos_sigma=((E1(KDX1).'-mean(yy))/error_pos)...
%                          *yy_cov*((E1(KDX1).'-mean(yy))/error_pos)';
% 
% fprintf('Poisson equation error: %g +- %g \n',...
%                                 error_pos,sqrt(error_pos_sigma));
% %
% 
tic;
for idx=1:100
    %H_p2
        xk2=xk2+ (deltat/2)*vk2;
        psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
        
        vk1=vk1+ (qk./mk).*(evE_mask( psik1-psik0, B3./(1j*KX2),KDX2 ) ...
            + deltat/2*vk2.*evE_mask(psik1, B3,~KDX2));
        vk3=vk3+ (-qk./mk).*(evE_mask( psik1-psik0, B1./(1j*KX2),KDX2 )...
            + deltat/2*vk2.*evE_mask(psik1, B1,~KDX2));
%          
        E2(KDX2)=E2(KDX2) + mean(bsxfun(@times, wk.*qk,...
            conj(psik1(:,KDX2) -psik0(:,KDX2)) )).'./(1j.*KX2(KDX2))...
            /L1/L2/L3;
        E2(~KDX2)= E2(~KDX2) - (deltat/2)*mean(bsxfun(@times, ...
            wk.*vk2.*qk, conj(psik1(:,~KDX2)))).'/L1/L2/L3;
         E2(~KDX)=0;
        psik0=psik1;
      
end
  toc/100
  
  
  round(26-randn(15,1)*4)
  
  
  