%% Dispersion relation for Landau Damping
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%
% Use MATLABs symbolic toolbox, and define
% frequency $\omega$, wave vector $k$
syms omega k x

% Set the number of significant decimal digits
% for variable-precision arithmetic
digits(16); 
numz=100;


% Set up grid of wave vectors the dispersion relation should be solved for
% kgrid=0.01:0.005:0.6;
% kgrid=0.01:0.01:0.5;
%kgrid=0.1:0.1:0.4;
kgrid=0.1:0.1:0.6;


%% Plasma dispersion function Z(x)
% The plasma dispersion function can be found in many textbooks
% 
% $$ Z(x)= \frac{1}{\sqrt{\pi}} \int_{\gamma} 
%    \frac{ \mathrm{e}^{-z^2}}{z-x} \mathrm{d}z$$
Z=symfun( sqrt(sym(pi))*exp(-x^2)*(1j-erfi(x)),x);



%% Dispersion relation
% The dispersion relation for the Vlasov-Poisson system with an initial state 
% Maxwellian (set $\alpha=1$)
% 
% $$f_0(t=0,x,v)=\frac{\alpha}{\sqrt{2 \pi} v_{th}}
%       \mathrm{e}^{-\frac{(v-v_0)^2 }{2 v_{th}^2 }} $$
%
% with thermal velocity $v_{th}$,
% mean velocity $v_0$ and plasma frequency $\omega_p$ is given as
%
% $$  D(\omega,k)= 1 + \alpha \left( \frac{\omega_p}{k v_{th}} \right)^2 
%  \left[1+ \frac{1}{\sqrt{2} v_{th} } \left( \frac{\omega}{k} - v_0 \right)
%   Z\left( \frac{1}{\sqrt{2} v_{th} } \left( \frac{\omega}{k} - v_0 \right)
%   \right) \right] $$.
%
% Set parameters
vth=1;    % Thermal velocity of electrons
omegap=1; % Plasma frequency
alpha=1;  % Normalization
v0=0;     % Mean velocity

% Dispersion relation in symbolic variables
D=symfun(1+ alpha*(omegap/vth/k)^2*(1+ (omega/k-v0)/vth/sqrt(2)*...
    Z((omega/k-v0)/vth/sqrt(2))), ...
    [omega,k]);


% Bump on tail
% Bump size
bz=0.1;
% maxwellian
vth1=1;
v1=0;
alpha1=(1-bz);
% bump
v2=4.5;
vth2=0.5;
alpha2=bz;

D=symfun(1+...
    alpha1*(omegap/vth1/k)^2*(1+ (omega/k-v1)/vth1/sqrt(2)*...
    Z((omega/k-v1)/vth1/sqrt(2)))...
    +alpha2*(omegap/vth2/k)^2*(1+ (omega/k-v2)/vth2/sqrt(2)*...
    Z((omega/k-v2)/vth2/sqrt(2))), ...
    [omega,k]);



%% Find zeros
% Use MATLABS internal variable precision arithmetic
% to solve for the zeroes of the dispersion relation for every k
OM=zeros(length(kgrid),numz); %contains omegas 
for idx=1:length(kgrid)
    for jdx=1:numz
        sol=[];
        % repeated search for zero until a solution is found
        while(isempty(sol))
           sol=vpasolve(D(omega,kgrid(idx))==0,'random',true);
        end
        OM(idx,jdx)=sol;
    end
end

numz=20;
boxsz=[-0.6,0.6,-2.5,4];
%% Find zeros
% Use MATLABS internal variable precision arithmetic
% to solve for the zeroes of the dispersion relation for every k
OM=zeros(length(kgrid),numz^2); %contains omegas 
for idx=1:length(kgrid)
    for jdx=1:numz
        for kdx=1:numz
        sol=[];
        % repeated search for zero until a solution is found
        while(isempty(sol))
           sol=vpasolve(D(omega,kgrid(idx))==0,...
       (boxsz(2)-boxsz(1))/numz*jdx +boxsz(1)+...
       1j*((boxsz(4)-boxsz(3))/numz*kdx +boxsz(3)));
        end
        OM(idx,jdx*numz+kdx)=sol;
        end
    end
end





%% Visualize result
figure;
plot(kgrid,real(OM),'*'); grid on;
xlabel('k'); ylabel('frequency');
title('zeros of dispersion relation')

figure;
plot(kgrid,-imag(OM),'*'); grid on;
xlabel('k'); ylabel('damping rate');
title('zeros of dispersion relation')

%% Standard test case $k=0.3$
% Found the following zeros for the dispersion relation for 
% the standard test case $k=0.5$
%
disp(sort(unique(OM(abs(kgrid-0.3)<1e-10,:).')))





