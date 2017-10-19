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
Np=1e5;    % Number of Lagrangian particles
Np_i=Np;
Np_e=Np;

Nf=3;      % Number of Fourier Modes
dt=0.05;   % Time step
dt_cov=5;  % time step for covariance

q=-1;      % charge
m_e=1;     % electron mass always normalized
T_e=1;    % Hot electrons
m_i=1038; % Ion mass
T_i=0.1;  % Ions are colder than electrons
c=1;      % Lightspeed relative to electron thermal velocity

% Default parameters
B0=0; alphar=0; alphai=0;v01=0;v02=0;eps=0;delta=0;epsi=0; betai=0; betar=0;
sigma1=1;sigma2=1;


%%
% Available test cases are:
%
% * landau - strong Landau damping
% * weibel - the Weibel instability
% * weibels - the streaming Weibel instability
%
testcase='em'; % 'weibel','landau','weibels'


switch (testcase)
    
    case('weibel')
        % Weibel instability
        eps=0; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
        betar=sign(q)*1e-3; betai=0;
        k=1.25;    % Wave vector
        sigma1=0.02/sqrt(2);
        sigma2=sqrt(12)*sigma1;
        v01=0;
        v02=0;
        delta=0;
        tmax=400;
        B0=0;
    case {'weibels','weibels2'}
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
        B0=0;
     case ('weibels1')
        % Symmetric Streaming Weibel instability
        sigma1=0.1/sqrt(2);
        sigma2=sigma1;
        k=0.2;
        betai=sign(q)*1e-3; betar=0;
        v01=0.3;
        v02=-0.3;
        delta=0.5;
        eps=0;
        tmax=400;
    case('landau')
        % Strong Landau damping
        eps=0.5;
        k=0.5;
        sigma1=1;sigma2=1;
        betar=0;betai=0;v01=0;v02=0;delta=0;
        tmax=45;
        B0=0;
    case('alfven')
        eps=0;
        k=pi;
        sigma1=1;sigma2=1;
        betar=0;betai=1;
        v01=0;v02=0;delta=0;
        tmax=45;
        B0=1;
        
    case('ion_acoustic')
        B0=0;
        k = 0.6283185 ;
        m_i=200; % Ion mass
        T_i=1e-4; Te=1;
        sigma1=1;sigma2=1;
        betar=0;betai=0;v01=0;v02=0;delta=0;
        eps=0;
        epsi=0.1;
        %alphai=0.2/k;
        Nf=1;
        dt=0.05;
        tmax=1000;
    case('em')
         % Electromagnetic wave
         c=10;
         k=0.4;
         tmax=10;   
         dt=0.001;
         betar=0.001;
         B0=0;
     case('xwave')
         % Exotic wave
         %eps=0.1;
         c=50;
         k=0.1;
         tmax=10;   
         dt=0.05;
         betar=0.1;
         B0=10;
      case ('magnetoacoustic')
        B0=2;
        k = 0.8;
        m_i=40; % Ion mass
        T_i=1e-4; T_e=1;
        c=10;
        epsi=0.1;
        dt=0.01;
        tmax=20;
        Nf=1;
end


% Length of domain
L=2*pi/k;
% Initial condition for electrons
f0e=@(x,v1,v2) (1+ eps*cos(k*x)).*exp(-v1.^2./sigma1^2/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))/2/sigma1/sigma2/pi;
% Initial condition for ions
sigmaio=sqrt(T_i/T_e.*m_e./m_i);
f0i=@(x,v1,v2) (1+ epsi*cos(k*x)).*exp(-v1.^2./sigmaio^2/2 -v2.^2/2/sigmaio.^2)/2/sigmaio/sigmaio/pi;


%% Initialize particles - Sampling
sbl=sobolset(3,'Skip',2);
sbl= scramble(sbl,'MatousekAffineOwen');
xk=sbl(1:Np*2,1); % Uniformly distributed U(0,1)
vk1=norminv(sbl(1:Np*2,2));  % Normally distributed $N(0,1)$
vk2=norminv(sbl(1:Np*2,3));

el_idx=1:Np;
ion_idx=(Np+1):2*Np;

%Importance sampling of electron and ion perturbation
uniformrnd=xk; eps_=ones(2*Np,1);%Sample electrons
eps_(el_idx)=eps;eps_(ion_idx)=epsi;
if (eps~=0 || epsi~=0)
    xk=uniformrnd*L;
    idx=0;
    while (idx<=15 ) %a bit much
        xk=xk- (xk + eps_.*sin(k*xk)/k -L*uniformrnd)./...
            (1+eps_.*cos(k*xk));
        idx=idx+1;
        xk=mod(xk,L);
    end
else
    xk=uniformrnd*L;
end
clear uniformrnd eps_;

%Sample electrons
vk1(el_idx)=vk1(el_idx)*sigma1;  % Normally distributed $N(0,\sigma)$
vk2(el_idx)=vk2(el_idx)*sigma2;

vk2(1:floor(delta*Np))=vk2(1:floor(delta*Np)) + v01;
vk2(floor(delta*Np)+1:end)=vk2(floor(delta*Np)+1:end) +v02;

g0e=@(x,v1,v2) (1+eps*cos(k*x)).*exp(-v1.^2./sigma1^2/2 ).*...
    (delta*exp(-(v2-v01).^2/2/sigma2.^2) + ...
    (1-delta)*exp( - (v2-v02).^2/2/sigma2^2))/2/sigma1/sigma2/pi/L;




% Sample ions
g0i=@(x,v1,v2) (1+epsi*cos(k*x)).*...
    exp(-v1.^2./sigmaio^2/2 -v2.^2/2/sigmaio.^2)/2/sigmaio/sigmaio/pi/L;
vk1(ion_idx)=vk1(ion_idx)*sigmaio;
vk2(ion_idx)=vk2(ion_idx)*sigmaio;

% Charge and mass for every particle
qk=[ones(Np,1)*q; ones(Np,1)*(-q)];
mk=[ones(Np,1)*m_e; ones(Np,1)*m_i];


% Write likelihoods
fk=[f0e(xk(el_idx),vk1(el_idx),vk2(el_idx));...
    f0i(xk(ion_idx),vk1(ion_idx),vk2(ion_idx))];  % Vlasov likelihood
gk=[g0e(xk(el_idx),vk1(el_idx),vk2(el_idx));...
    g0i(xk(ion_idx),vk1(ion_idx),vk2(ion_idx))];  % Particle (Sampling) likelihood


%% Initialize Particle in Fourier Maxwell Solver
% Spatial wave vector
kx=(2*pi)/L*(1:Nf)';
% Particle weight
wk=fk./gk;

% wk(el_idx)=wk(el_idx)-(mean(wk(el_idx))-L);
% wk(ion_idx)=wk(ion_idx)-(mean(wk(ion_idx))-L);
%
% vk1(ion_idx)=vk1(ion_idx)-mean(vk1(ion_idx).*wk(ion_idx))./mean(wk(ion_idx));
% vk2(ion_idx)=vk2(ion_idx)-mean(vk2(ion_idx).*wk(ion_idx))./mean(wk(ion_idx));
%
% vk1(el_idx)=vk1(el_idx)-mean(vk1(el_idx).*wk(el_idx))./mean(wk(el_idx));
% vk2(el_idx)=vk2(el_idx)-mean(vk2(el_idx).*wk(el_idx))./mean(wk(el_idx));

wk=wk.*2; %factor 2 because of mean with Np and two species

% Fourier Coefficients of the Fields
E1=zeros(length(kx),1);
E2=zeros(length(kx),1);
B=zeros(length(kx),1);

%%
% Splitting constants for composition methods. Set to scalar 1
% if only Strang splitting is desired
comp_gamma=[1/(2-2^(1/3)), -2^(1/3)/(2-2^(1/3)),  1/(2-2^(1/3))];
% comp_gamma=1;

%% Initializing Maxwell Solver
% Initial condition on k-th Fourier mode of B
B(kx==k)=betar + 1j*betai;
B(kx==-k)=betar - 1j*betai;
% %Initial condition on k-th Fourier mode of E_1
E2(kx==k)=alphar + 1j*alphai;
E2(kx==-k)=alphar - 1j*alphai;

% Solve poisson equation for intialization
E1=mean(bsxfun(@times,qk.*wk, exp(-1j*xk*kx.') )).'./(1j*kx)/L;
E1_0=0; E2_0=0;

% Allocate diagnostic output
Nt=ceil(tmax/dt);
Epot=zeros(Nt,2);Ekin=Epot;Bpot=zeros(Nt,1);Momentum=Epot;
Nt_cov=ceil(tmax/dt_cov);
rho_e_var=zeros(Nt_cov,1);rho_i_var=rho_e_var;
j_i_var=rho_e_var;j_e_var=rho_e_var;


% Evaluate mode and it's complex conjugate at once
evF=@(a,b) (real(a)*real(b) - imag(a)*imag(b)).*2.0;

% Determine scatter plot limits
vmax1=max(abs(vk1)); vmax2=max(abs(vk2));
runtime=tic;
psik0=exp(1j*xk*kx.');
for tdx=1:Nt
    t=tdx*dt;
    % Composition steps of the second order Strang splitting
    for splitdx=1:length(comp_gamma)
        deltat=comp_gamma(splitdx)*dt;
        
        %H_E
        %psik0=exp(1j*xk*kx.');
        %psik0=psik1;
        vk1=vk1 + (deltat/2)*(evF(psik0,E1)+E1_0).*qk./mk;
        vk2=vk2 + (deltat/2)*(evF(psik0,E2)+E2_0).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        %H_B
        E2=E2  -c^2*(deltat/2).*(1j.*kx).*B;
        
        %H_p2
        vk1=vk1 +  (deltat/2)*(evF(psik0,B)+B0).*vk2.*qk./mk;
        E2=E2  - (deltat/2)*mean(bsxfun(@times,...
            wk.*vk2.*qk, conj(psik0))).'/L;
        E2_0=E2_0 - (deltat/2)*mean(wk.*qk.*vk2)./L;
        
        % Center of the Strang splitting, therefore 2*deltat/2
        %H_p1
        xk=xk+ deltat*vk1;
        psik1=exp(1j*xk*kx.');
        
        % diagnose variance
        if (mod(t,dt_cov)==0)
            tdx_cov=t/dt_cov;
            % Integrated variance of E for the poisson equation
            yy=(bsxfun(@times,qk.*wk, psik1 ))./(1j*kx.')/L;
            rho_e_var(tdx_cov)=sum(var(yy( el_idx ,:)/2));
            rho_i_var(tdx_cov)=sum(var(yy( ion_idx ,:)/2));
            % Integrated variance of E for the Ampere equation
            yy=(bsxfun(@times, wk.*qk, conj(psik1-psik0) ))./1j./kx.'/L;
            j_e_var(tdx_cov)=sum(var(yy( el_idx ,:)/2))+...
                var((deltat)*wk(el_idx).*qk(el_idx).*vk1(el_idx)/2);
            j_i_var(tdx_cov)=sum(var(yy( ion_idx ,:)/2))+...
                var((deltat)*wk(ion_idx).*qk(ion_idx).*vk1(ion_idx)/2);
        end
        
        
        vk2=vk2 -  (evF(psik1-psik0,(B./1j./kx) ) - deltat*B0   ) .*qk./mk;
        E1=E1 + mean(bsxfun(@times, wk.*qk,...
            conj(psik1-psik0) )).'./1j./kx/L;
        E1_0=E1_0 - (deltat)*mean(wk.*qk.*vk1);
        
        %H_p2
        vk1=vk1+ (deltat/2)*(evF(psik1,B)+B0).*vk2.*qk./mk;
        E2=E2  - (deltat/2)*mean(bsxfun(@times, ...
            wk.*vk2.*qk, conj(psik1))).'/L;
        E2_0=E2_0 - (deltat/2)*mean(wk.*qk.*vk2)./L;
        
        %H_B
        E2=E2  -c^2*(deltat/2).*(1j.*kx).*B;
        
        %H_E
        vk1=vk1 + (deltat/2)*(evF(psik1,E1)+E1_0).*qk./mk;
        vk2=vk2 + (deltat/2)*(evF(psik1,E2)+E2_0).*qk./mk;
        B=B -(deltat/2).*(1j.*kx).*E2;
        
        xk=mod(xk,L);
        psik0=psik1;
    end
    
    %% Scatter plot of Lagrangian particles
    if (mod(t,tmax/10)==0)
        %         scatter3(xk,vk1,vk2,4,fk,'filled')
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
    
    
    Epot(tdx,1) =(E1'*E1)*(L*2)/2 + E1_0^2*L/2;
    Epot(tdx,2) =(E2'*E2)*(L*2)/2+ E2_0^2*L/2;
    Ekin(tdx,1) = mean(vk1.^2.*wk.*mk)/2;
    Ekin(tdx,2) = mean(vk2.^2.*wk.*mk)/2;
    Momentum(tdx,1)=mean(vk1.*wk.*mk)+ ( E2'*B+ B'*E2 + E2_0*B0 )*L;
    Momentum(tdx,2)=mean(vk2.*wk.*mk)-( E1'*B+ B'*E1+ E1_0*B0  )*L;
    Bpot(tdx)=(B'*B)*(L*2)/2;
end
runtime=toc(runtime);



set(0,'DefaultLegendFontSize',16,...
    'DefaultLegendFontSizeMode','manual',...
    'DefaultAxesFontSize', 14)
set(0,'DefaultLineMarkerSize', 10);
set(0,'DefaultLineLinewidth', 1);

prefix=['./plots/',testcase,'_multi_'];
%Save a file for later
save([prefix,'result.mat']);

%% Field Energys
time=0:dt:tmax-dt;
figure;
semilogy(time,Epot(:,end:-1:1),'-.','Linewidth',2); hold on;
semilogy(time,Bpot,'Linewidth',2)
xlabel('time'); grid on; axis tight;
% axis tight;
title('electromagnetic energy');
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


%% Variance
set(0,'DefaultLegendFontSize',16,...
    'DefaultLegendFontSizeMode','manual',...
    'DefaultAxesFontSize', 14)
time_cov=(0:dt_cov:tmax-dt_cov).';
figure;
semilogy(time_cov,rho_e_var,'Linewidth',3); hold on;
plot(time_cov,rho_i_var,'-.','Linewidth',3); grid on;
legend('electrons', 'ions');
xlabel('time');
axis([0,max(time_cov),-inf,inf])
title('IVAR[E_1] from Poisson');
print('-dpng',[prefix, 'ivar_E1_poisson.png'])

figure;
semilogy(time_cov,j_e_var,'Linewidth',3); hold on;
plot(time_cov,j_i_var,'-.','Linewidth',3);
legend('electrons', 'ions');
xlabel('time');
axis([0,max(time_cov),-inf,inf]); grid on;
title('IVAR[E_1] from Ampere');
print('-dpng',[prefix, 'ivar_E1_ampere.png'])

figure;
title('IVAR[E_1] species contribution');
semilogy(time_cov,rho_e_var./rho_i_var,'Linewidth',3);hold on;
plot(time_cov,j_e_var./j_i_var,'Linewidth',3);
legend('Poisson', 'Ampere');
xlabel('time');
ylabel('ions v.s. electrons'); grid on;
axis([0,max(time_cov),-inf,inf]);
print('-dpng',[prefix, 'ivar_E1_ratios.png'])


%Kinetic energy by species
% figure


% rho_e_var(tdx_cov)=sum(var(yy( el_idx ,:)/2));
%            rho_i_var(tdx_cov)=sum(var(yy( ion_idx ,:)/2));
%            % Integrated variance of E for the Ampere equation
%            yy=(bsxfun(@times, wk.*qk, conj(psik1-psik0) ))./1j./kx.'/L;
%            j_e_var(tdx_cov)=sum(var(yy( el_idx ,:)/2));
%            j_i_var(tdx_cov)=sum(var(yy( ion_idx ,:)/2));


% %% Kinetic Energy
% figure;
% semilogy(time,Epot,'-.','Linewidth',2); hold on;
% semilogy(time,Bpot,'Linewidth',2)
% xlabel('time'); grid on;
% % axis tight;
% title('electrostatic energy');
% l=legend('$\frac{1}{2} || E_1 ||^2$','$\frac{1}{2} || E_2 ||^2$',...
%     '$\frac{1}{2} || B ||^2$',...
%     'Location','SouthEast');
% set(l,'Interpreter','LaTeX','FontSize',16)
% print('-dpng',[prefix,'fieldenergy.png'])

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
yy_cov_e=cov(yy( el_idx ,:)/2)/Np; % Sample covariance for electrons
yy_cov_i=cov(yy(ion_idx,:)/2)/Np;

yy_cov=yy_cov_e + yy_cov_i;
% L2 error on the Fourier modes
error_pos=norm(E1.'-mean(yy)) ;
% Linearized propagation of uncertainty
error_pos_sigma=((E1.'-mean(yy))/error_pos)...
    *yy_cov*((E1.'-mean(yy))/error_pos)';

fprintf('Poisson equation error: %g +- %g \n',...
    error_pos,sqrt(error_pos_sigma));

fprintf('Charge density variance: el.=%g, ion=%g \n',...
    sum(diag(yy_cov_e) ),sum(diag(yy_cov_i) ))

% Variance in current is a bit more involved
psik1=exp(1j*(xk+ deltat*vk1)*kx.');
yy=(bsxfun(@times, wk.*qk, conj(psik1-psik0) ))./1j./kx.'/L;
yy_cov_e=cov(yy( el_idx ,:)/2)/Np; % Sample covariance for electrons
yy_cov_i=cov(yy(ion_idx,:)/2)/Np;
fprintf('Current density variance: el.=%g, ion=%g \n',...
    sum(diag(yy_cov_e) ),sum(diag(yy_cov_i) ))


% Output for paper
fprintf(' %03.2g\n',...
    sqrt(j_e_var(end)/Np),sqrt(j_i_var(end)/Np),...
    sqrt(rho_e_var(end)/Np),sqrt(rho_i_var(end)/Np),...
    error_pos)



% j_e=mean(yy( el_idx ,:)/2)
% j_i=mean(yy( ion_idx ,:)/2)

%         figure;
%         scatter3(xk(el_idx),vk1(el_idx),vk2(el_idx),4,fk(el_idx),'filled')
%         colorbar;
%         axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
%         xlabel('x');ylabel('v_1');zlabel('v_2');
%         title(sprintf('t=%g,',t))
%         drawnow; hold off;
%
%
%               figure;
%         scatter3(xk(ion_idx),vk1(ion_idx),vk2(ion_idx),4,fk(ion_idx),'filled')
%         colorbar;
%         axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
%         xlabel('x');ylabel('v_1');zlabel('v_2');
%         title(sprintf('t=%g,',t))
%         drawnow; hold off;


%
%          scatter(vk1(el_idx),vk2(el_idx),4,fk(el_idx),'filled')
%          colorbar;
% %          axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
%          xlabel('x');ylabel('v_1');zlabel('v_2');
%          title(sprintf('t=%g,',t))
%          drawnow; hold off;


%         scatter3(xk,vk1,vk2,4,fk,'filled')
%         colorbar;
%         axis([0,L,-vmax1,vmax1,-vmax2,vmax2])
%         xlabel('x');ylabel('v_1');zlabel('v_2');
%         title(sprintf('t=%g,',t))
%         drawnow; hold off;

