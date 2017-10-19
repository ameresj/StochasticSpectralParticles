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
Np=1e5;  %1e5  % Number of Lagrangian particles
Nf=3;    %3   % Number of Fourier Modes
dt=0.5;   %0.5 % Time step


q=-1;   % charge
m=1;    % mass
mu0=1;   %Vacuum permeability


% splitting or midpoint rule with picard or newton iteration
pusher='picard'; % split, picard, newton
% Gauss Legendre points
numGL=0;  %analytic integration for 0, random for -1




%%
% Available test cases are:
%
% * landau - strong Landau damping
% * weibel - the Weibel instability
% * weibels - the streaming Weibel instability
%
testcase='weibels'; % 'weibel','landau','weibels'


switch (testcase)
    
    case('weibel')
        % Weibel instability
        eps=0.05; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
        betar=sign(q)*1e-3; betai=0;
        k=1.25;    % Wave vector
        sigma1=0.02/sqrt(2);
        sigma2=sqrt(12)*sigma1;
        v01=0;
        v02=0;
        delta=0;
        tmax=400; %100
        
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
        tmax=400;
        
    case('landau')
        % Strong Landau damping
        eps=0.5;
        k=0.5;
        sigma1=1;sigma2=1;
        betar=0;betai=0;v01=0;v02=0;delta=0;
        tmax=10;
end


% Length of domain
L=2*pi/k;
% Initial condition
f0=@(x,v1,v2) (1+ eps*cos(k*x)).*exp(-v1.^2./sigma1^2/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))/2/sigma1/sigma2/pi;

% Gauss legendre points
[lgwt_x,lgwt_w]=lgwt(abs(numGL),0,1);


%% Initialize particles - Sampling
sbl=sobolset(4,'Skip',2);
xk=sbl(1:Np,1)*L; % Uniformly distributed U(0,L)
alphak=sbl(1:Np,4); % integration
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


vk1=vk1-mean(vk1.*wk)/Np;
vk2=vk2-mean(vk2.*wk)/Np;

% Fourier Coefficients of the Fields
E1=zeros(length(kx),1);
E2=zeros(length(kx),1);
B=zeros(length(kx),1);

%%
% Splitting constants for composition methods. Set to scalar 1
% if only Strang splitting is desired
comp_gamma=[1/(2-2^(1/3)), -2^(1/3)/(2-2^(1/3)),  1/(2-2^(1/3))];
%comp_gamma=1;


%% Initializing Maxwell Solver
% Initial condition on k-th Fourier mode of B
B(kx==k)=betar + 1j*betai;
B(kx==-k)=betar - 1j*betai;

% Solve poisson equation for intialization
E1=mean(bsxfun(@times,qk.*wk, exp(-1j*xk*kx.') )).'./(1j*kx)/L;

% Allocate diagnostic output
Nt=ceil(tmax/dt);
Epot=zeros(Nt,2);Ekin=Epot;Bpot=zeros(Nt,1);Momentum=Epot;
iterations=zeros(Nt,1);
% Evaluate mode and it's complex conjugate at once
evF=@(a,b) (real(a)*real(b) - imag(a)*imag(b)).*2.0;

% Determine scatter plot limits
vmax1=max(abs(vk1)); vmax2=max(abs(vk2));
runtime=tic;
for tdx=1:Nt
    t=tdx*dt;
    % Composition steps of the second order Strang splitting
    for splitdx=1:length(comp_gamma)
        deltat=comp_gamma(splitdx)*dt;
        
        %H_E
        psik0=exp(1j*xk*kx.');
        
        vk1=vk1 + (deltat/2)*evF(psik0,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evF(psik0,E2).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        
        %H_B
        E2=E2-(deltat/2).*(1j.*kx).*B;
        
        % Center of the Strang splitting, therefore 2*deltat/2
        
        switch(pusher)
            case 'split'
                %H_p2
                vk1=vk1 +  (deltat/2)*evF(psik0,B).*vk2.*qk./mk;
                E2=E2  - mu0*(deltat/2)*mean(bsxfun(@times,...
                    wk.*vk2.*qk, conj(psik0))).'/L;
                %H_p1
                
                xk=xk+ deltat*vk1;
                psik1=exp(1j*xk*kx.');
                
                
                vk2=vk2 -  (evF(psik1-psik0,(B./1j./kx) )  ) .*qk./mk;
                E1=E1 + mean(bsxfun(@times, wk.*qk,...
                    conj(psik1-psik0) )).'./1j./kx/L;
                
                
                %H_p2
                vk1=vk1+ (deltat/2)*(evF(psik1,B)).*vk2.*qk./mk;
                E2=E2  - (deltat/2)*mean(bsxfun(@times, ...
                    wk.*vk2.*qk, conj(psik1))).'/L;
                
            case {'picard','newton'}
                tol=5e-16;
                %Explicit euler initial guess
                xkt=xk;
                Bxt=evF(psik0,B).*qk./mk;
                vk1t= vk1+ deltat*Bxt.*vk2;
                vk2t= vk2- deltat*Bxt.*vk1;
                error=inf;
                num=0;
                
                while (error >tol)
                    %psik1=exp(1j*(xk + deltat/8*(3*vk1+vk1t)  )*kx.');
                    psik1=exp(1j*(xk + deltat/4*(vk1+vk1t)  )*kx.');
                    Bxt2=evF(psik1,B).*qk./mk; %B(x(deltat/2,0))
                    
                    if strcmp(pusher,'picard')
                        %Picard iterations
                        vk1t_new = vk1 + deltat*( vk2+vk2t)/2.*Bxt2;
                        vk2t_new = vk2 - deltat*( vk1+vk1t)/2.*Bxt2;
                    else
                        %Newton iterations
                        DBxt2=evF(psik1,B.*(1j.*kx)).*qk./mk;
                        
                        % vk1t_new=((- 8.*vk1.*Bxt2.^2 - 8.*vk1t.*Bxt2.^2).*deltat.^2 ...
                        %     + (32.*vk2.*Bxt2).*deltat + 32.*vk1 - 32.*vk1t)./...
                        %     ((2.*vk2.*DBxt2 + 2.*vk2t.*DBxt2 - 8.*Bxt2.^2).*deltat.^2 +...
                        %     (- vk1.*Bxt2.*DBxt2 - vk1t.*Bxt2.*DBxt2).*deltat.^3 - 32);
                        % vk2t_new=((2.*DBxt2.*vk1t.^2 - 2.*DBxt2.*vk2.^2 ...
                        %     - 2.*DBxt2.*vk1.^2 + 2.*DBxt2.*vk2t.^2 ...
                        %     - 8.*Bxt2.^2.*vk2 ...
                        %     - 8.*Bxt2.^2.*vk2t).*deltat.^2 + ...
                        %     (-32.*vk1.*Bxt2).*deltat + 32.*vk2 - 32.*vk2t)./...
                        %     ((2.*vk2.*DBxt2 + 2.*vk2t.*DBxt2 - 8.*Bxt2.^2).*deltat.^2 + ...
                        %     (- vk1.*Bxt2.*DBxt2 - vk1t.*Bxt2.*DBxt2).*deltat.^3 - 32);
                        
                        
                        vk1t_new=((- 4*vk1.*Bxt2.^2 - 4*vk1t.*Bxt2.^2)*deltat^2 + ...
                            (16*vk2.*Bxt2.*deltat) + 16*vk1 - 16.*vk1t)./...
                            ((2*vk2.*DBxt2 + 2*vk2t.*DBxt2 - 4*Bxt2.^2).*deltat.^2 + ...
                            (- vk1.*Bxt2.*DBxt2 - vk1t.*Bxt2.*DBxt2).*deltat.^3 - 16);
                        vk2t_new=((2.*DBxt2.*vk1t.^2 - 2.*DBxt2.*vk2.^2 - 2.*DBxt2.*vk1.^2 + ...
                            2.*DBxt2.*vk2t.^2 - 4.*Bxt2.^2.*vk2 - 4.*Bxt2.^2.*vk2t).*deltat^2 + ...
                            (-16.*vk1.*Bxt2).*deltat + 16.*vk2 - 16.*vk2t)./((2.*vk2.*DBxt2 + 2.*vk2t.*DBxt2 -...
                            4.*Bxt2.^2)*deltat.^2 + (- vk1.*Bxt2.*DBxt2 - vk1t.*Bxt2.*DBxt2).*deltat^3 - 16);
                        vk2t_new=vk2t-vk2t_new;
                        vk1t_new=vk1t-vk1t_new;
                    end
                    
                    error=max( [abs(vk1t_new-vk1t);abs(vk2t_new-vk2t)]);
                    vk1t=vk1t_new;vk2t=vk2t_new;
                    num=num+1;
                    
                end
                iterations(tdx)=num;
                %                 fprintf('%d \n', num)
                xk0=xk;
                xk=xk + deltat*(vk1+vk1t)/2;
                psik1=exp(1j*xk*kx.');
                
                if (numGL==0)
                    %Exact integration
                    E1=E1 + mu0*mean(bsxfun(@times, wk.*qk,...
                        conj(psik1-psik0) )).'./1j./kx/L;
                    E2=E2  + mu0*mean(bsxfun(@times, wk.*qk.*(vk2+vk2t)./(vk1+vk1t),...
                        conj(psik1-psik0) )).'./1j./kx/L;
                elseif(numGL>0)
                    
                    %With Gauss legendre weights
                    tau=lgwt_x*deltat;
                    tau_w=lgwt_w*deltat;
                    
                    for kdx=1:length(kx)
                        E1(kdx)=E1(kdx)- mu0*mean((exp(-1j*kx(kdx)*(xk0 + (vk1+vk1t)/2*tau.' ))*tau_w)...
                                            .*wk.*qk.*(vk1+vk1t)/2)/L;
                        E2(kdx)=E2(kdx)- mu0*mean((exp(-1j*kx(kdx)*(xk0 + (vk1+vk1t)/2*tau.' ))*tau_w)...
                                            .*wk.*qk.*(vk2+vk2t)/2)/L;
                        % E1(kdx)=E1(kdx)- mu0*mean(((exp(-1j*kx(kdx)*(xk0 +vk1*tau' + ...
                        %     (vk1t-vk1)/deltat/2*tau'.^2)   ).*...
                        %     (vk1 + (vk1t-vk1)/deltat*tau'))*tau_w).*wk.*qk)/L;
                        % E2(kdx)=E2(kdx)- mu0*mean(((exp(-1j*kx(kdx)*(xk0 +vk1*tau' +...
                        %     (vk1t-vk1)/deltat/2*tau'.^2) ).*...
                        %     (vk2 + (vk2t-vk2)/deltat*tau'))*tau_w).*wk.*qk)/L;
                    end
                    
                else
                    % Random
                    %alphak=0.5;
                    
                    Ns=abs(numGL); %number of strata
                    
                    for ds=0:(Ns-1)
                        
                    psik0=exp(-1j*(xk0 + (vk1+vk1t)/2.*(alphak +ds)/Ns*deltat )*kx.' );
                     E1=E1 - mu0*deltat*mean(bsxfun(@times,psik0,...
                              wk.*qk.*(vk1+vk1t)/2)).'/L/Ns;
                    E2=E2 - mu0*deltat*mean(bsxfun(@times,psik0,...
                           wk.*qk.*(vk2+vk2t)/2)).'/L/Ns;
                    end
                end
                vk1=vk1t;
                vk2=vk2t;
                
%                 numGL=10
%                  Eu1=mu0*mean(bsxfun(@times, wk.*qk,...
%                         conj(psik1-psik0) )).'./1j./kx/L;
%                  [lgwt_x,lgwt_w]=lgwt(numGL,0,1);
%                  tau=lgwt_x*deltat;
%                  tau_w=lgwt_w*deltat;
%                     
%                     for kdx=1:length(kx)
%                         Eu2(kdx)=- mu0*mean((exp(-1j*kx(kdx)*(xk0 + (vk1+vk1t)/2*tau' ))*tau_w)...
%                                             .*wk.*qk.*(vk1+vk1t)/2)/L;
%                     end
%                  
%                  Eu1
%                  Eu2(:)
                
                    
        end
        
        %H_B
        E2=E2-(deltat/2).*(1j.*kx).*B;
        
        %H_E
        vk1=vk1 + (deltat/2)*evF(psik1,E1).*qk./mk;
        vk2=vk2 + (deltat/2)*evF(psik1,E2).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        xk=mod(xk,L);
    end
    
    
    %% Scatter plot of Lagrangian particles
    if (mod(t,tmax/10)==0)
        %         scatter3(xk,vk1,vk2,8,fk,'filled')
        %         colorbar;
        %         axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
        %         xlabel('x');ylabel('v_1');zlabel('v_2');
        %         title(sprintf('t=%g,',t))
        %         drawnow; hold off;
        
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
    Momentum(tdx,1)=mean(vk1.*wk.*mk)+ ( E2'*B+ B'*E2 )*L;
    Momentum(tdx,2)=mean(vk2.*wk.*mk)-( E1'*B+ B'*E1  )*L;
    Bpot(tdx)=(B'*B)*(L*2)/2;
end
runtime=toc(runtime);

set(0,'DefaultLegendFontSize',16,...
    'DefaultLegendFontSizeMode','manual',...
    'DefaultAxesFontSize', 14)
set(0,'DefaultLineMarkerSize', 10);
set(0,'DefaultLineLinewidth', 1);


prefix=['./plots/',testcase,'_',pusher,'_'];
if numGL~=0
 prefix=sprintf('%s%d_',prefix,numGL);
end

%% Field Energys
time=0:dt:tmax-dt;
figure;
semilogy(time,Epot(:,end:-1:1),'-.','Linewidth',2); hold on;
semilogy(time,Bpot,'Linewidth',2)
xlabel('time'); grid on; axis tight;
% axis tight;
title('electrostatic energy');
l=legend('$\frac{1}{2} || E_2 ||^2$','$\frac{1}{2} || E_1 ||^2$',...
    '$\frac{1}{2} || B ||^2$',...
    'Location','SouthEast');
set(l,'Interpreter','LaTeX','FontSize',16)

% Growth rate
% φ(z) 0.02784.= ex
print('-dpng',[prefix,'fieldenergy.png'])

%% Energy Error
figure;
energy=sum(Epot,2)+Bpot+sum(Ekin,2);
semilogy(time, abs((energy-energy(1)))/energy(1),'Linewidth',1) ;hold on;
grid on; axis tight;
title('relative energy error');
xlabel('time');
ylabel('$|\mathcal{H}(t)-\mathcal{H}(0)|/\mathcal{H}(0)$',...
    'interpreter','latex')
print('-dpng',[prefix, 'energy_error.png'])
%% Momentum Error
figure;
semilogy(time, abs(Momentum-Momentum(2,:)))
legend('P_1','P_2','Location','SouthEast');
xlabel('time');
title('absolute momentum error');
axis tight; grid on;
print('-dpng',[prefix, 'momentum_error.png'])

%% Performance Statistics
% For one stage of the integrator
fprintf('Overall time: %g s\n',runtime);
fprintf('Pushed %g particles/s \n', Np*Nt/runtime*2*length(comp_gamma));

figure;
plot(time, iterations); grid on;
xlabel('time');
ylabel(sprintf('number of iterations'));

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

% %Save a file for later
energy_error=abs((energy-energy(1)))/energy(1);
save([prefix,'result.mat'],'time','Epot','Bpot','iterations','runtime',...
    'pusher','Ekin','error_pos','error_pos_sigma','Np','energy','energy_error','Momentum');
%
close all;
disp(error_pos)

% close all;
%
% error_pos

% %Derivation of newton method
%
%
% syms vk1 vk2 vk1t vk2t xk B(x) x deltat;
%
%
% F=[vk1 + deltat*( vk2+vk2t)/2*B(xk + deltat/8*(3*vk1+vk1t))-vk1t;
%     vk2 - deltat*( vk1+vk1t)/2.*B(xk + deltat/8*(3*vk1+vk1t))-vk2t];
% Fp=simplify(jacobian(F,[vk1t,vk2t]),10);
% collect(simplify(Fp\F,10),[deltat.^2,deltat])
% syms a b c d e k x real
%
% simplify(int( (a+ b*x)*exp(-1j*k*(c + d*x + e*x^2)),x))
%
%
%
%
%  syms v20 v2t tau t k v10 v10 v1t x0 real
% 
% 
%  dd=simplify(int( (v20 + (v2t - v20)*tau/t)*exp(-1j*k*(x0+ (v1t-v10)/t*tau   )),tau,0,t),10);


