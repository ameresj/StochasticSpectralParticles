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

%% Solver parameters
Np=5e4;     % Number of particles
Nf=16;     % Number of finite element basis functions
degree=3;  % B-Spline degree
rungekutta_order=3; %Order of the runge Kutta time integrator
dt=0.1;


%% Landau Damping parameters
eps=0.05; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
k=0.5;    % Wave vector
L=2*pi/k; % length of domain
qm=-1;    % negative charge to mass = electrons, $\frac{q}{m}=-1$


% Initial condition
f0=@(x,v) (1+ eps*cos(k*x)).*exp(-0.5.*v.^2)/sqrt(2*pi);
f_equ=@(v) exp(-0.5.*v.^2)/sqrt(2*pi);
tmax=20;


%% Control variate
h=[]; % Empty for full f, requires much more particles, e.g. Np=1e6
% h=@(x,v) exp(-0.5.*v.^2)/sqrt(2*pi); %maxwellian for delta f
%h=f0 % initial condition for delta f



%% Initialize particles - Sampling
xk=rand(Np,1)*L; % Uniformly distributed U(0,L)
vk=randn(Np,1)*1 + 0;  % Normally distributed N(0,1)

g0=@(x,v) exp(-0.5.*v.^2)/sqrt(2*pi)/L; %Initial sampling density

fk=f0(xk,vk);       % Vlasov likelihood
gk=g0(xk,vk);       % Particle (Sampling) likelihood


%% Initialize B-Spline Finite Element Poisson Solver
fem.knots=linspace(0,L,Nf+1); % Knots for B-Splines
fem.dx=fem.knots(2)-fem.knots(1);
fem.pp=bspline(fem.knots(1:degree+2));


% Plot first basis function $\psi_1(x)$
x=linspace(min(fem.pp.breaks),max(fem.pp.breaks));
figure('Name','Basis Function','Numbertitle','off');
plot(x,ppval(fem.pp,x));
axis([x(1),x(end),-inf,inf]); grid on;
xlabel('x'); ylabel('basis function');
title('B-Spline Basis Function');

% Finite element matrices
% Dervative of Testfunction
fem.pp_prime=fnder(fem.pp,1);

fem.Krow=zeros(degree+1,1);
fem.Mrow=zeros(degree+1,1);
for idx=1:degree+1
    offset=(idx-1)*fem.dx;
    % L2 projection
    fem.Mrow(idx)=integral...
        ( @(x) ppval(fem.pp,x).*ppval(fem.pp,x-offset ),...
        offset, max(fem.pp_prime.breaks));
    
    % Poisson Matrix
    fem.Krow(idx)=integral...
        ( @(x) ppval(fem.pp_prime,x).*ppval(fem.pp_prime,x-offset ),...
        offset, max(fem.pp_prime.breaks));
end

% Assemble circulant matrices
% Mass matrix
first_row=zeros(Nf,1);
first_row(1:degree+1)=fem.Mrow;
first_row(end:-1:end-degree+1)=fem.Mrow(2:end);
fem.M=sparse(gallery('circul',first_row)); %Circulant matrix from first row
% Poisson Matrix - weak laplace
first_row=zeros(Nf,1);
first_row(1:degree+1)=fem.Krow;
first_row(end:-1:end-degree+1)=fem.Krow(2:end);
fem.K=sparse(gallery('circul',first_row));

%Determine right hand side for control variate h if present
if isempty(h)
    rhs_h=0;
else
    fprintf('Assembling right hand side for control variate\n')
    fprintf('This is very general, and will take some time!\n')
    rhs_h=zeros(Nf,1);
    for idx=1:Nf
        offset=(idx-1)*fem.dx;
         %Choose large tolerances to speed up computation
         %Integral psi(x) h(x,v) dxdv 
         rhs_h(idx)=integral2...
             (@(x,v)ppval(fem.pp,x-offset).*h(mod(x,L),v),...
             0+offset,fem.dx*(degree+1)+offset,...
             -inf,inf,...
             'method','iterated',...
             'RelTol',1e-2,...
             'AbsTol',1e-3);
    end
    fprintf('Done\n');
end

%% Set coefficients for symplectic Runge Kutta
switch rungekutta_order
    case 1
        rksd=1; %Symplectic Euler
        rksc=1;
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

wk=fk./gk;
xk(Np/2+1:Np)=xk(1:Np/2);
vk(Np/2+1:Np)=vk(1:Np/2);
gk(Np/2+1:Np)=gk(1:Np/2);
wk(1:Np/2)=fk(1:Np/2)./gk(1:Np/2);
        wk(Np/2+1:end)=-f_equ(vk(Np/2+1:end))./gk(Np/2+1:end);
        wk=wk*2;

%Allocate output
cv=struct('alpha',[],'rho',[]); %Control variate diagnostic

rhs_cv=0;

tstep=1;
figure('Name','Phase Space Density','Numbertitle','off');
for t=0:dt:tmax
    fprintf('t=%08.3f, %05.2f%%\n ', t, t/tmax*100)
    
    xk=mod(xk,L); 


    
    %Loop over all stages of the Runge Kutta method
    for rksidx=1:length(rksd)
%         
%         %% Particle Weight for field solve
%         if isempty(h)
%             %full f = no control variate
%             wk=fk./gk;
%             rhs_cv=0;
%         else
%             %delta f with control variate
%             hk=h(xk,vk);
%             
%             %Test moment
%             psik=(xk-L/2).^2; %sin(xk./2*pi./L), xk^3
%             
%             %Determine correlation
%             sigma=cov([fk./gk.*psik,hk./gk.*psik]);
%             %Optimization coefficient
%             cv.alpha(tstep)=sigma(2,2:end)\sigma(1,2:end);
%             %Correlation coefficient
%             cv.rho(tstep)=sigma(1,2)./sqrt(prod(diag(sigma)));
%             wk=(fk-cv.alpha(tstep)*hk)./gk;
%             
%             rhs_cv=cv.alpha(tstep)*rhs_h;
%         end
        
        
       %% Determine psi_k, all testfunctions,
        %  evaluated for all particles, results in a sparse matrix        
        %cell number for every particle
        [~, particle_cell]=histc(xk, fem.knots);
        
        % go to local coordinates in cell
        xx=mod(xk,fem.dx);
        
        row=repmat( (1:Np), degree+1,1).'; % Row index  (Particle)
        col=zeros(Np,degree+1);            % Column index (Basis function)
        val=zeros(Np,degree+1);
        %Evaluate every piece of the piecewise polynomial basis function
        for kdx=0:degree
            col(:, kdx+1)=mod((particle_cell-1-kdx),Nf)+1;
            val(:, kdx+1)=polyval(fem.pp.coefs(kdx+1,:),xx).*wk;
        end
        
        %psik=sparse(row, col,val,Np, Nf ); %Includes also particle weight
        %rhs=mean(psik).'; % Mean over all basis functions (weighted)
        %rhs_cov=cov(psik); %Covariance
        
        % Fast accumulation, for mean
        rhs=accumarray([reshape(col,numel(col),1); (1:Nf)'], ...
            [reshape(val,numel(val),1); zeros(Nf,1)])./Np ...
            +rhs_cv;
        
        
        
        rho=fem.M\rhs; %L2 projection for charge density
        
        %Poisson solve
        fem.K_fourier=fft(full(fem.K(1,:))).';
        phi=fft(full(rhs),[],1)./fem.K_fourier;
        phi(1,:)=0; %offset
        phi=real(ifft(phi,'symmetric'));
        
        %Diagnostic
        if (rksidx==1)
            kineticenergy(tstep)=0.5*mean(vk.^2.*fk./gk);
            fieldenergy(tstep)=(qm).^2*0.5*phi'*fem.K*phi;
            momentum(tstep)=mean(vk.*fk./gk);
            l2norm(tstep)=mean(fk.^2./gk);
            
        end
        
%         if (rksidx==1 && mod(tstep-1,ph_hist.freq)==0)
%             %% Phase space histogram
%             [~,cellx]=histc(xk,ph_hist.x); %bin particles to cells
%             [~,cellv]=histc(vk,ph_hist.v); % and get the indicies
%             %particles out of range are remapped, but not discarded
%             cellv=min(max(1,cellv), ph_hist.numv);
%             
%             %Accumulate weight in boxes and normalize
%             phasespace=accumarray( [cellx,cellv], wk,...
%                 [ph_hist.numx, ph_hist.numv])./phi_hist.vol./Np;            
%             %Add contribution of control variate
%             if (~isempty(h))
%                phasespace=phasespace+...
%                    cv.alpha(tstep)*(h(ph_hist.xx,ph_hist.vv));
%             end
%             
%             %subtract background
%             if ~isempty(ph_hist.background)
%                 phasespace=phasespace-...
%                     ph_hist.background(ph_hist.xx,ph_hist.vv);
%             end
%             
%             %update limits for color axis
%             phi_hist.limits=[min(min(min(phasespace)),phi_hist.limits(1)),....
%                              max(max(max(phasespace)),phi_hist.limits(2))];
%             
%             
%             pcolor(ph_hist.x(1:end-1),ph_hist.v(1:end-1),...
%                 phasespace.');
%             xlabel('x');ylabel('v');
%             caxis(phi_hist.limits);
%             %scatter(xk,vk,5,fk,'filled'); colorbar;
%             shading interp; 
%             title(sprintf('t=%08.3f', t));
%             colormap jet; colorbar;
%             drawnow;
%         end
        
        
        
        %% Determine d/dx psi_k, gradient of testfunctions
        %Evaluate every piece of the piecewise polynomial basis function
        Ex=zeros(Np,1);
        for kdx=0:degree
            col=mod((particle_cell-1-kdx),Nf)+1;
            Ex=Ex + polyval(fem.pp_prime.coefs(kdx+1,:),xx).*phi(col);
        end
        Ex(Np/2+1:Np)=0;
        vk=vk +  rksc(rksidx)*dt*Ex*(qm);
        xk=xk +  rksd(rksidx)*dt*vk;
        xk=mod(xk,L);
        
        
    end
    tstep=tstep+1;
end


%% Discussion
time=0:dt:t;
figure('Name','Electrostatic Energy','Numbertitle','off');
semilogy(time, fieldenergy);
xlabel('time'); grid on;
ylabel('electrostatic energy');

% Include decay rate for linear landau damping WITHOUT COLLISIONS
if (eps<0.1 && k==0.5)
hold on;
% Obtain zero of dispersion relation with dispersion_landau.m
omega=1.415661888604536 - 0.153359466909605i;
plot(time,0.5*fieldenergy(1)*abs(exp(-1j*omega*(time-0.4))).^2);
% linear analysis with frequency
plot(time,0.5*fieldenergy(1)*real(exp(-1j*omega*(time-0.4))).^2);
legend('numerical', 'linear analysis', 'linear analysis');

hold off;
end



% figure('Name','Kinetic Energy','Numbertitle','off');
% semilogy(time, kineticenergy);
% xlabel('time'); grid on;
% ylabel('kinetic energy');



% figure('Name','Momentum','Numbertitle','off');
% plot(time, momentum);
% xlabel('time'); grid on;
% ylabel('momentum');

% 
% %For higher spline degree, the spectral fidelity of the basis functions
% %will increase, and therefore decrease the momentum error
% figure('Name','Momentum Error','Numbertitle','off');
% semilogy(time, abs(momentum-momentum(1)));
% xlabel('time'); grid on;
% ylabel('absolute momentum error');


% %The energy error depends on the monte carlo noise, controlled by
% %the particle number, the time integrator, and the time step
% figure('Name','Energy Error','Numbertitle','off');
% energy=fieldenergy+kineticenergy;
% semilogy(time, abs((energy-energy(1))/energy(1)));
% xlabel('time'); grid on;
% ylabel('error');
% title('relative energy error');
% 



%Variance reduction by the control variate
if (~isempty(h))
figure('Name','Correlation Coefficient','Numbertitle','off');
plot(time,cv.rho)
xlabel('time'); grid on;
ylabel('correlation coefficient \rho^2');
title('Variance reduction by a factor of ...');
end







