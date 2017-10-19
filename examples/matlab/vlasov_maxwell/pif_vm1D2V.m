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
dt=0.5;   % Time step


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
        eps=1e-3; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
        betar=sign(q)*1e-3; betai=0;
        k=1.25;    % Wave vector
        sigma1=0.02/sqrt(2);
        sigma2=sqrt(12)*sigma1;
        v01=0;
        v02=0;
        delta=0;
        tmax=400;
        tmax=150;
    case ('weibels')
        % Streaming Weibel instability
        sigma1=0.1/sqrt(2);
        sigma2=sigma1;
        k=0.2;
        betai=sign(q)*1e-3; betar=0;
        v01=0.5;
        v02=-0.1;
        delta=1/6.0;
        eps=0;
        tmax=150;
        
    case('landau')
        % Strong Landau damping
        eps=0.5;
        k=0.5;
        sigma1=1;sigma2=1;
        betar=0;betai=0;v01=0;v02=0;delta=0;
        tmax=40;
end
tmax=50;

% Length of domain
L=2*pi/k; 
% Initial condition 
f0=@(x,v1,v2) (1+ eps*cos(k*x)).*exp(-v1.^2./sigma1^2/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))/2/sigma1/sigma2/pi;



%% Initialize particles - Sampling
sbl=sobolset(3,'Skip',2);
xk=sbl(1:Np,1)*L; % Uniformly distributed U(0,L)

vk1=norminv(sbl(1:Np,2))*sigma1;  % Normally distributed $N(0,\sigma)$
vk2=norminv(sbl(1:Np,3))*sigma2;

vk2(1:floor(delta*Np))=vk2(1:floor(delta*Np)) + v01;
vk2(floor(delta*Np)+1:end)=vk2(floor(delta*Np)+1:end) +v02;

g0=@(x,v1,v2) exp(-v1.^2./sigma1^2/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))/2/sigma1/sigma2/pi/L;

% Charge and mass for every particle
qk=ones(Np,1)*q;
mk=ones(Np,1)*m;

fk=f0(xk,vk1,vk2);       % Vlasov likelihood
gk=g0(xk,vk1,vk2);       % Particle (Sampling) likelihood

%% Initialize Particle in Fourier Maxwell Solver
% Spatial wave vector
kx=(2*pi)/L*(1:Nf)';
% Particle weight
wk=fk./gk;


% Fourier Coefficients of the Fields
E1=zeros(length(kx),1);
E2=zeros(length(kx),1);
B=zeros(length(kx),1);

%%
% Splitting constants for composition methods. Set to scalar 1
% if only Strang splitting is desired
comp_gamma=[1/(2-2^(1/3)), -2^(1/3)/(2-2^(1/3)),  1/(2-2^(1/3))];
comp_gamma=1;
%% Initializing Maxwell Solver
% Initial condition on k-th Fourier mode of B
B(kx==k)=betar + 1j*betai;
B(kx==-k)=betar - 1j*betai;

% Solve poisson equation for intialization
E1=mean(bsxfun(@times,qk.*wk, exp(-1j*xk*kx.') )).'./(1j*kx)/L;

% Allocate diagnostic output
Nt=ceil(tmax/dt);
Epot=zeros(Nt,2);Ekin=Epot;Bpot=zeros(Nt,1);Momentum=Epot;

% Evaluate mode and it's complex conjugate at once
evF=@(a,b) (real(a)*real(b) - imag(a)*imag(b)).*2.0;

% Determine scatter plot limits
vmax1=max(abs(vk1)); vmax2=max(abs(vk2));
runtime=tic;
psik0=exp(1j*xk*kx.'); %Initialize first field evaluation
for tdx=1:Nt
    t=tdx*dt;
    % Composition steps of the second order Strang splitting
    for splitdx=1:length(comp_gamma)
        deltat=comp_gamma(splitdx)*dt;
        
        %H_E
        %psik0=exp(1j*xk*kx.');
        
        vk1=vk1 + (deltat/2)*evF(psik0,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evF(psik0,E2).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        
        %H_B
        E2=E2  -(deltat/2).*(1j.*kx).*B;
        
        %H_p2
        vk1=vk1 +  (deltat/2)*evF(psik0,B).*vk2.*qk./mk;
        E2=E2  - mu0*(deltat/2)*mean(bsxfun(@times,...
                                    wk.*vk2.*qk, conj(psik0))).'/L;
        
        % Center of the Strang splitting, therefore 2*deltat/2
        %H_p1
        xk=xk+ deltat*vk1;
        psik1=exp(1j*xk*kx.');
        
        vk2=vk2 -  evF(psik1-psik0,(B./1j./kx) ).*qk./mk;
        E1=E1 + mu0*mean(bsxfun(@times, wk.*qk,...
                                conj(psik1-psik0) )).'./1j./kx/L;
        
        
        %H_p2
        vk1=vk1+ (deltat/2)*evF(psik1,B).*vk2.*qk./mk;
        E2=E2  - mu0*(deltat/2)*mean(bsxfun(@times, ...
                                        wk.*vk2.*qk, conj(psik1))).'/L;
        
        %H_B
        E2=E2-(deltat/2).*(1j.*kx).*B;

        %H_E
        vk1=vk1 + (deltat/2)*evF(psik1,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evF(psik1,E2).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        xk=mod(xk,L);
        psik0=psik1;
    end
    
    
    %% Scatter plot of Lagrangian particles
    if (mod(t,tmax/10)==0)
%         scatter3(xk,vk1,vk2,8,fk,'filled')
%         colorbar;
%         axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
%         xlabel('x');ylabel('v_1');zlabel('v_2');
%         title(sprintf('t=%g,',t))
%         drawnow; hold off;
%         
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
    
    
    Epot(tdx,1) =(E1'*E1)*(L*2)  /2;
    Epot(tdx,2) =(E2'*E2)*(L*2)/2;
    Ekin(tdx,1) = mean(vk1.^2.*wk)/2;
    Ekin(tdx,2) = mean(vk2.^2.*wk)/2;
    
    Momentum(tdx,1)=mean(vk1.*wk)+ ( E2'*B+ B'*E2  )*L;
    Momentum(tdx,2)=mean(vk2.*wk)-( E1'*B+ B'*E1  )*L;
    Bpot(tdx)=(B'*B)*(L*2)/2;
end
runtime=toc(runtime);

%% Field Energys
time=0:dt:tmax-dt;
figure;
semilogy(time,Epot); hold on;
semilogy(time,Bpot)
xlabel('time'); grid on;
title('electromagnetic energy');
l=legend('$\frac{1}{2} || E_1 ||^2$','$\frac{1}{2} || E_2 ||^2$',...
    '$\frac{1}{2} || B ||^2$',...
    'Location','SouthEast');
set(l,'Interpreter','LaTeX','FontSize',16)

%% Energy Error
figure
energy=sum(Epot,2)+Bpot+sum(Ekin,2);
semilogy(time, abs((energy-energy(1)))/energy(1)) ;hold on;
grid on;
axis tight;
title('relative energy error');
xlabel('time');
ylabel('$|\mathcal{H}(t)-\mathcal{H}(0)|/\mathcal{H}(0)$',...
    'interpreter','latex')

%% Momentum Error
figure;
semilogy(time, abs(Momentum-Momentum(1,:)))
legend('P_1','P_2'); grid on;
xlabel('time'); 
title('absolute momentum error');
axis tight

%% Performance Statistics
% For one stage of the integrator
fprintf('Overall time: %g s\n',runtime);
fprintf('Pushed %g particles/s \n', Np*Nt/runtime*2*length(comp_gamma));



%% Poisson equation
% The Poisson equation is conserved by the Hamiltonian splitting. This can 
% be checked here. In order to check the vadility of the solution the
% variance on the field $E_1$ can be calculated.
% 
% % Samples
yy=(bsxfun(@times,qk.*wk, exp(-1j*xk*kx.') ))./(1j*kx.')/L;
yy_cov=cov(yy)/Np; % Sample covariance

% L2 error on the Fourier modes
error_pos=norm(E1.'-mean(yy)) ;
% Linearized propagation of uncertainty
error_pos_sigma=((E1.'-mean(yy))/error_pos)...
                        *yy_cov*((E1.'-mean(yy))/error_pos)';

fprintf('Poisson equation error: %g +- %g \n',...
                                error_pos,sqrt(error_pos_sigma));

