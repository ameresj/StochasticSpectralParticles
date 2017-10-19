%% PFC 2D Vlasov-Poisson Solver
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%% Features
%
% * Positive Flux conservative scheme
% * *Symplectic* (for Vlasov-Poisson) splitting



%% Landau Damping parameters
eps=0.05; % Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
kx=0.5;    % Wave vector
L=2*pi/kx; % length of domain
qm=-1;    % negative charge to mass = electrons, $\frac{q}{m}=-1$

% Initial condition
f0=@(x,v) (1+eps*cos(kx*x))./sqrt(2*pi).*exp(-0.5.*v.^2);
% Background to be subtracted from the phase space plot
background=@(x,v) 1./sqrt(2*pi).*exp(-0.5.*v.^2);
% External electric field
E_ext=@(x,t) 0.*x+0.*t;

dt=0.1; % Time step
tmax=50;
rungekutta_order=3; %Order of the runge Kutta time integrator


Nx=32; % Number of cells in spatial direction
Nv=32; % Number of cells in velocity direciton
vmax=4.5;
vmin=-4.5;

%% Define phase space mesh
v=linspace(vmin,vmax,Nv+1).';v=v(1:end-1);
x=linspace(0,L,Nx+1).';x=x(1:end-1);

[XX,VV]=ndgrid(x,v);



f=f0(XX,VV);





