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
Np=5e5;     % Number of particles
Np_diag=2e3; % Number of particles to keep for diagnostic reasons
Nf=32;     % Number of finite element basis functions
degree=3;  % B-Spline degree
rungekutta_order=3; %Order of the runge Kutta time integrator
dt=0.5;
tmax=100;

%% Bump-on-Tail parameters
eps=0.03; % Amplitude of perturbation
modes=1;  % Mode number = number of vorticies
k=0.3;    % Wave vector
L=2*pi/k*modes; % length of domain
qm=-1;    % negative charge to mass = electrons, $\frac{q}{m}=-1$

nb=0.1;        %Bump size
v0=4.5;        %Mean velocity of the Bump
bumpsigma=0.5;

%Nakamura, Yabe
% Initial condition
f0=@(x,v)(1+eps*cos(k*x+pi/4))./sqrt(2*pi).*...
           ((1-nb).*exp(-0.5.*v.^2) +...
           nb.*exp(-0.5.*(v-v0).^2/bumpsigma.^2)./bumpsigma);
vmin=-4;
vmax=7;
            
        

%% Fokker-Planck
% Collision frequency
fokker_planck.theta=0; % weak 1e-3, strong 0.1, none 0
% Additional parameters for equilibrium
% if empty, then determined by simulation for momentum
% and kinetic energy conservation
fokker_planck.mu=[];     %0
fokker_planck.sigma=[];  %1*sqrt(2*fokker_planck.theta) 

%% Nearest neighbour coarse graining
coarse_grain.freq=1e4;     %Frequency
coarse_grain.neighbours=5; %number of nearest neighbours to grain


%% Control variate
%h=[]; % Empty for full f, requires much more particles, e.g. Np=1e6
 h=@(x,v) (1/(1+bumpa)).*...
                 (exp(-0.5.*v.^2) + bumpa/bumpsigma*...
                 exp(-0.5.*(v-v0).^2/bumpsigma.^2))/sqrt(2*pi);
%h=f0 % initial condition for delta f
h=[];

%% Phase space histogram
% Parameters
% Background that is subtracted in the plot
% set to empty [] if no background is supplied
ph_hist.background=[]; 
ph_hist.numx=60; %Number of cells in x
ph_hist.numv=60; %Number of cells in v
ph_hist.freq=10; %Frequency of the time consuming plot,
                  %Note: the binning is cheap, but the graphics expensive
% Initialization
ph_hist.x=linspace(0,L,ph_hist.numx+1);  %Bin edges can be nonuniform
ph_hist.v=linspace(vmin,vmax,ph_hist.numv+1);
[ph_hist.xx,ph_hist.vv]=ndgrid(ph_hist.x(1:end-1),ph_hist.v(1:end-1)); % mesh
%Determine cell volume
phi_hist.vol=diff(ph_hist.x)'*diff(ph_hist.v);
%Set initial value for phase space
phasespace=f0(ph_hist.xx,ph_hist.vv); 
if ~isempty(ph_hist.background) %subtract background if defined
    phasespace=phasespace-ph_hist.background(ph_hist.xx,ph_hist.vv);
end
%Define limits for adaptavive colorbar
phi_hist.limits=[min(min(phasespace)), max(max(phasespace))];


%% Initialize particles - Sampling
% Randomized Quasi-Monte-Carlo Numbers. The randomization is necesseary
% to get a reliable estimate for the control variate
sobolnum = sobolset(2,'Skip',0);
sobolnum= scramble(sobolnum,'MatousekAffineOwen'); %Scrambling=Randomization

xk=sobolnum(1:Np,1)*L; % Uniformly distributed U(0,L)
vk=sobolnum(1:Np,2)*(vmax-vmin)+vmin;  % Uniformly distributed U(vmin,vmax)

g0=@(x,v) 1.0.*(v>=vmin & v<=vmax)./(vmax-vmin)/L+0.*x; %Initial sampling density

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



%Allocate output
cv=struct('alpha',[],'rho',[]); %Control variate diagnostic

% Test particles
particle_diag=struct('x',[],'v',[],'Phi',[],'f',[],'g',[],'time',[]);


tstep=1;
figure('Name','Phase Space Density','Numbertitle','off');
for t=0:dt:tmax
    fprintf('t=%08.3f, %05.2f%%\n ', t, t/tmax*100)
    
    xk=mod(xk,L); 

    
    %Loop over all stages of the Runge Kutta method
    for rksidx=1:length(rksd)
        
        %% Particle Weight for field solve
        if isempty(h)
            %full f = no control variate
            wk=fk./gk;
            rhs_cv=0;
        else
            %delta f with control variate
            hk=h(xk,vk);
            
            %Test moment
            psik=(xk-L/2).^2; %sin(xk./2*pi./L), xk^3
            
            %Determine correlation
            sigma=cov([fk./gk.*psik,hk./gk.*psik]);
            %Optimization coefficient
            cv.alpha(tstep)=sigma(2,2:end)\sigma(1,2:end);
            %Correlation coefficient
            cv.rho(tstep)=sigma(1,2)./sqrt(prod(diag(sigma)));
            wk=(fk-cv.alpha(tstep)*hk)./gk;
            
            rhs_cv=cv.alpha(tstep)*rhs_h;
        end
        
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
            [reshape(val,numel(val),1); zeros(Nf,1)])./Np...
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
            %Kullback-Leibler-Entropy relative to initial condition
            kl_entropy(tstep)=mean(fk./gk.*log(fk./f0(xk,vk)));
            %Shannon entropy
            sh_entropy(tstep)=mean(fk.*log(fk)./gk);
            
        end
        
        if (rksidx==1 && mod(tstep-1,ph_hist.freq)==0)
            %% Phase space histogram
            [~,cellx]=histc(xk,ph_hist.x); %bin particles to cells
            [~,cellv]=histc(vk,ph_hist.v); % and get the indicies
            %particles out of range are remapped, but not discarded
            cellv=min(max(1,cellv), ph_hist.numv);
            
            %Accumulate weight in boxes and normalize
            phasespace=accumarray( [cellx,cellv], wk,...
                [ph_hist.numx, ph_hist.numv])./phi_hist.vol./Np;            
            %Add contribution of control variate
            if (~isempty(h))
               phasespace=phasespace+...
                   cv.alpha(tstep)*(h(ph_hist.xx,ph_hist.vv));
            end
            
            %subtract background
            if ~isempty(ph_hist.background)
                phasespace=phasespace-...
                    ph_hist.background(ph_hist.xx,ph_hist.vv);
            end
            
            %update limits for color axis
            phi_hist.limits=[min(min(min(phasespace)),phi_hist.limits(1)),....
                             max(max(max(phasespace)),phi_hist.limits(2))];
            
            
            pcolor(ph_hist.x(1:end-1),ph_hist.v(1:end-1),...
                phasespace.');
            xlabel('x');ylabel('v');
            caxis(phi_hist.limits);
            shading interp; 
            title(sprintf('t=%08.3f', t));
            colormap jet; colorbar;
            drawnow;
        end
        
        %% Extract test particles for post processing
        %
        % Determine Potential energy of every particle
        if (rksidx==1)
            particle_diag(tstep).time=t;
            particle_diag(tstep).x=xk(1:Np_diag);
            particle_diag(tstep).v=vk(1:Np_diag);
            particle_diag(tstep).f=fk(1:Np_diag);
            particle_diag(tstep).g=gk(1:Np_diag);
            Phik=zeros(Np_diag,1);
            for kdx=0:degree
                col=mod((particle_cell(1:Np_diag)-1-kdx),Nf)+1;
                Phik=Phik + polyval(fem.pp.coefs(kdx+1,:),xx(1:Np_diag)).*phi(col);
            end
            particle_diag(tstep).Phi=Phik;
        end
        
        
        
        
        %% Determine d/dx psi_k, gradient of testfunctions
        %Evaluate every piece of the piecewise polynomial basis function
        Ex=zeros(Np,1);
        for kdx=0:degree
            col=mod((particle_cell-1-kdx),Nf)+1;
            Ex=Ex + polyval(fem.pp_prime.coefs(kdx+1,:),xx).*phi(col);
        end
        vk=vk +  rksc(rksidx)*dt*Ex*(qm);
        xk=xk +  rksd(rksidx)*dt*vk;
        xk=mod(xk,L);
        
    
        
        %% Fokker Planck collisions
        if (fokker_planck.theta>0)
            mass=mean(fk./gk);
            
            fp.theta=fokker_planck.theta;
            %Determine the local equilibrium if needed
            %Mean velocity
            if isempty(fokker_planck.mu)
                fp.mu=mean(fk./gk.*vk)./mass;
            else
                fp.mu=fokker_planck.mu;
            end
            %standard deviation of the local maxwellian in v
            if isempty(fokker_planck.sigma)
                fp.sigma=sqrt(var(fk./gk.*vk./mass))...
                    *sqrt(2*fokker_planck.theta);
            else
                fp.sigma=fokker_planck.sigma;
            end
            
            %Define Fokker-Planck Equilibrium
            fp.feq=@(v) sqrt(fp.theta/pi)/fp.sigma...
                *exp(-fp.theta.*((v-fp.mu)./fp.sigma).^2);
            %Save equilibrium
            fp.feq_old=fp.feq(vk);
            
            %Propagate samples through Ornstein Uhlenbeck process
            vk=vk.*exp(-fp.theta*dt) ...
                + fp.mu*(1-exp(-fp.theta*dt)) ...
                + sqrt(fp.sigma.^2/(2*fp.theta).*...
                ( 1-exp(-2*fp.theta.*dt))) ...
                .*randn(Np,1);
            
            %Rescale weight
            gk=gk.*(fp.feq(vk)./fp.feq_old);
            fk=fk.*(fp.feq(vk)./fp.feq_old); fp.feq_old=[];
        end
        
       %% Coarse graining
        if mod(tstep,coarse_grain.freq)==0
            fprintf('Coarse graining particles \n');
         
          [neighbour_index] = knnsearch...
                 ([xk,vk], [xk,vk],...
                 'K', coarse_grain.neighbours+1,...
                 'distance', 'euclidean');
           gk(neighbour_index(:,1))=mean(gk(neighbour_index),2);
           fk(neighbour_index(:,1))=mean(fk(neighbour_index),2);
           neighbour_index=[];
        end        
        
    end
    tstep=tstep+1;
end

figure;
for idx=1:length(particle_diag)
    plot(particle_diag(idx).x,particle_diag(idx).v,'.');
    drawnow;
    
end



fprintf('Assembling transfer operator.\n');
mat=0;
for idx=1:length(particle_diag)
        
    %Sum L^2
%     mat=mat...
%         +(mod(bsxfun(@minus, particle_diag(idx).x,particle_diag(idx).x.')...
%                           + L/2, L)- L/2).^2 ...
%         +bsxfun(@minus, particle_diag(idx).v ,particle_diag(idx).v.').^2;

%with weights
    mat=mat + (...
        +(mod(bsxfun(@minus, particle_diag(idx).x,particle_diag(idx).x.')...
                          + L/2, L)- L/2).^2 ...
        +bsxfun(@minus, particle_diag(idx).v ,particle_diag(idx).v.').^2);
end



[U,S,V]=svds(mat,10);
S=diag(S)
plot(S)
% 
% num=2;
% 
% [U,S,V]=svds(mat,num);
% S=diag(S)
%             fig=figure('Name','coher_singulars','Numbertitle','off');
%             semilogy(S(2:end));
%             ylabel('singular value');
%             xlabel('no. of singular value');
%             fig=0;
            num=3;
for idx=1:1:length(particle_diag)
    scatter(particle_diag(idx).x,particle_diag(idx).v,15,V(:,num),'filled');
    colormap jet;
    colorbar;
    axis([0, L,vmin,vmax]);
    title(sprintf('t=%4.4f',particle_diag(idx).time));
    xlabel('x');ylabel('v');
    drawnow;
end

close all;
for th=0:0.01:1
v=[particle_diag.v];
x=[particle_diag.x];
Phi=[particle_diag.Phi];
%Define weight
w=[particle_diag.f]./[particle_diag.g];

index=V(:,num)<=min(V(:,num)).*th;

x=x(index,:);
w=w(index,:);
v=v(index,:);
Phi=Phi(index,:);
%     scatter(x(:,50),v(:,50),30,V(index,num),'filled');

Ekin=0.5*mean(v.^2.*w);
Epot=mean( Phi.*w)/2;
mass=mean(w);

% Energy=mean(0.5*(v.^2 +Phi).*w);
plot(th,var(Ekin),'*'); hold on;
end
% plot(time,0.5*mean(v.^2.*w))
% figure;
% plot(time,mean( [particle_diag.Phi].*w)/2)




%select one vortex
V(:,num);





%% Discussion
time=0:dt:t;
figure('Name','Electrostatic Energy','Numbertitle','off');
semilogy(time, fieldenergy);
xlabel('time'); grid on;
ylabel('electrostatic energy');
% 
% % Include decay rate for linear landau damping WITHOUT COLLISIONS
% if (eps<0.1 && k==0.5)
% hold on;
% % Obtain zero of dispersion relation with dispersion_landau.m
% omega=1.415661888604536 - 0.153359466909605i;
% plot(time,0.5*fieldenergy(1)*abs(exp(-1j*omega*(time-0.4))).^2);
% % linear analysis with frequency
% plot(time,0.5*fieldenergy(1)*real(exp(-1j*omega*(time-0.4))).^2);
% legend('numerical', 'linear analysis', 'linear analysis');
% 
% hold off;
% end
% 
% 
% 
% figure('Name','Kinetic Energy','Numbertitle','off');
% semilogy(time, kineticenergy);
% xlabel('time'); grid on;
% ylabel('kinetic energy');
% 
% 
% 
% figure('Name','Momentum','Numbertitle','off');
% plot(time, momentum);
% xlabel('time'); grid on;
% ylabel('momentum');
% 
% 
% %For higher spline degree, the spectral fidelity of the basis functions
% %will increase, and therefore decrease the momentum error
% figure('Name','Momentum Error','Numbertitle','off');
% semilogy(time, abs(momentum-momentum(1)));
% xlabel('time'); grid on;
% ylabel('absolute momentum error');
% 
% 
% %The energy error depends on the monte carlo noise, controlled by
% %the particle number, the time integrator, and the time step
% figure('Name','Energy Error','Numbertitle','off');
% energy=fieldenergy+kineticenergy;
% semilogy(time, abs((energy-energy(1))/energy(1)));
% xlabel('time'); grid on;
% ylabel('error');
% title('relative energy error');
% 
% 
% %Kullback-Leibler-Entropy relative to initial condition
% figure('Name','Kullback-Leibler-Entropy','Numbertitle','off');
% plot(time,kl_entropy)
% xlabel('time'); grid on;
% ylabel('entropy');
% title('Kullback-Leibler-Entropy relative to f(t=0)');
% 
% 
% %Shannon-Entropy 
% figure('Name','Shannon-Entropy of f','Numbertitle','off');
% plot(time,sh_entropy)
% xlabel('time'); grid on;
% ylabel('entropy of f');
% title('Shannon-Entropy of f');
% 
% 
% %Variance reduction by the control variate
% if (~isempty(h))
% figure('Name','Correlation Coefficient','Numbertitle','off');
% plot(time,cv.rho)
% xlabel('time'); grid on;
% ylabel('correlation coefficient \rho^2');
% title('Variance reduction by a factor of ...');
% end
% 
% 
% 
% 
% 


