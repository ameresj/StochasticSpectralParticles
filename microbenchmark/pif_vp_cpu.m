%% Particle in Cell 2D for Vlasov-Poisson-Fokker-Planck
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%% Features
%
% * Arbitrary degree *B-Spline Finite Elements*
% * *Symplectic* (for Vlasov-Poisson) Runge Kutta *time integrator*
% * Control Variate for Field Solve
% * Fokker-Planck collisions by *Ornstein-Uhlenbeck process*
% * Nearest neighbour *coarse graining*
% * Phase space density reconstruction by histogram
%

%% Equations
% This code solves the Vlasov-Poisson system coupled to linear
% Fokker-Planck. The Vlasov Fokker-Planck equation reads
%
% $$\frac{ \partial f}{\partial t} +
% v \cdot \nabla_x f + \frac{q}{m} (E) \cdot \nabla_v f
%  -\theta \frac{\partial}{\partial v} \left( (v-\mu) f \right) -
% \frac{\sigma^2}{2} \frac{\partial^2 f}{\partial v^2} = 0 $$
%
% coupled to the Poisson equation for the potential $\Phi$
%
% $$  -\Delta\Phi = \rho - 1, \qquad  E:=-\nabla \Phi $$
%

% maxNumCompThreads(4)
%% Solver parameters
Np=1e5;     % Number of particles
Nf=64;     % Number of finite element basis functions
degree=3;  % B-Spline degree
rungekutta_order=3; %Order of the runge Kutta time integrator
dt=0.1;
tmax=15;

%% Bump-on-Tail parameters
eps=0.1; % Amplitude of perturbation
k=0.5;    % Wave vector
L=2*pi/k; % length of domain
qm=-1;    % negative charge to mass = electrons, $\frac{q}{m}=-1$

% Initial condition
f0=@(x,v)(1+eps*cos(k*x)).*exp(-0.5.*v.^2)./sqrt(2*pi);


%% Initialize particles - Sampling
xk=rand(Np,1)*L; % Uniformly distributed U(0,L)
vk=randn(Np,1);

g0=@(x,v) exp(-0.5.*v.^2)./sqrt(2*pi)/L;

% Vlasov likelihood
% Particle (Sampling) likelihood
wk=f0(xk,vk)./g0(xk,vk);


kx=((1:Nf)*2*pi/L)';


%% Set coefficients for symplectic Runge Kutta
switch rungekutta_order
    case 1
        rksd=[1, 0]; %Symplectic Euler
        rksc=[0, 1];
    case 2
        rksd=[0.5, 0.5 ];
        rksc=[0, 1];
    case 3
        rksd=[2/3, -2/3, 1  ];
        rksc=[ 7/24, 3/4, -1/24];
    case 4
        rk4sx=real((2^(1/3) +2^(-1/3)-1)/6);
        rksd=[ 2*rk4sx +1 , -4*rk4sx-1, 2*rk4sx+1, 0];
        rksc=[ rk4sx + 0.5 , -rk4sx, -rk4sx, rk4sx +0.5];
end



fieldenergy_pif=zeros(tmax/dt+1,1);

pift=tic;

tstep=1;
rho=zeros(Nf,1);
E=zeros(Nf,1);
for t=0:dt:tmax
    %     fprintf('t=%08.3f, %05.2f%%\n ', t, t/tmax*100)
    
    %Loop over all stages of the Runge Kutta method
    for rksidx=1:length(rksd)
        xk=mod(xk,L);
        
        
        for kdx=1:Nf
            rho(kdx)=sum(exp(-1j*xk*(kdx*2*pi/L)  ).*wk )/L/Np;
        end
        
        E=-rho./(1j*kx);
        
        for kdx=1:Nf
            psik=exp(1j*xk*(kdx*2*pi/L));
            vk=vk +  rksc(rksidx)*dt*2*...
                qm*(real(psik)*real(E(kdx))-imag(psik)*imag(E(kdx)));
        end
        
        xk=xk +  rksd(rksidx)*dt*vk;
        
        
        %Diagnostic
        if (rksidx==1)
            
            fieldenergy_pif(tstep)=(qm).^2*0.5*real(E'*E)*2*L;
        end
    end
    tstep=tstep+1;
end
time_pif=toc(pift);
fprintf('PIF: %g \n', time_pif);
%
% %% Discussion
%  time=0:dt:t;
% figure('Name','Electrostatic Energy','Numbertitle','off');
%  semilogy(time, fieldenergy_pif);
%
%
%
%
% xlabel('time'); grid on;
% ylabel('electrostatic energy');
% legend('pic','pif');

