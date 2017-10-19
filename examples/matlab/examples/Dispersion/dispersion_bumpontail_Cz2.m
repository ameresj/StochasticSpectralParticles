%% Dispersion relation for Landau Damping
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%
% Use MATLABs symbolic toolbox, and define
% frequency $\omega$, wave vector $k$
syms omega k x

% Set the number of significant decimal digits
% for variable-precision arithmetic
digits(64); 

% Set up grid of wave vectors the dispersion relation should be solved for
% kgrid=0.01:0.005:0.6;
% kgrid=0.01:0.01:0.5;
kgrid=0.1:0.1:0.5;

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

%% Problem Parameters
% Use MATLABS internal variable precision arithmetic to evaluate the
% Dispersion relation on the contour of a circle in the complex
% plane with radius _radius_.
% For all
% $ k \in$ _kgrid_ we search for
% $\omega \in C$ with 
% $|\omega| \leq$ _radius_.
% 
radius=3; % 
n=128;    % Number of unit roots, increase for precision
K=n/2-1;  % Number of zeroes to retrieve

% Roots of unity in symbolic expression
zk=vpa(exp(1j*2*sym('pi')*(1:n)/n));

OMEGA=zeros(length(kgrid),K); % Roots of the Dispersion relation
RES=zeros(length(kgrid),K);   % Residual for Postprocessing
% Set the default residual to infinity in order to distinguish non zeros
RES(:,:)=Inf;

for idx=1:length(kgrid)

    % Print status
    fprintf('k= %08.4f,  %05.2f%%\n',...
    kgrid(idx), idx/length(kgrid)*100);   


% gk=double(vpa(1./D(zk*radius,kgrid(idx))));
% [r,a,b,mu,nu,w] = ratdisk(gk, n-1-K, K);

 %% Compute Zeroes and Poles by Cauchy Integrals
 % Algorithm [Cz2] taken from 
 % 
 % 
 s = ifft(double(vpa(1./D( zk*radius, kgrid(idx) ) )));
 H = hankel(s(2:K+1), s(K+1:2*K));
 H2 = hankel(s(3:K+2), s(K+2:2*K+1)); 
 w = eig(H2,H);

 % 
 w=w*radius;
 
 %% 
 % Evaluate residual
 % $|D(\omega,k)|$
 %
 res=abs(vpa(D(w,kgrid(idx))));
 
 % Sort by residual
 [res,I]=sort(abs(res));
 w=w(I);

 RES(idx,1:length(res))=res(:);
 OMEGA(idx,1:length(w))=w(:);
 

end

%% Visualize result
% The roots $\omega$ are approximations, but sorted according to the 
% residual for every $k$. We take the first _Nz_ roots. 
% 
% 
Nz=4;
w=OMEGA(:,1:Nz);

figure;
plot(kgrid,real(w),'*'); grid on;
xlabel('k'); ylabel('frequency');
title('zeros of dispersion relation')

figure;
plot(kgrid,-imag(w),'*'); grid on;
xlabel('k'); ylabel('damping rate');
title('zeros of dispersion relation')

%% Standard test case $k=0.5$
% Found the following zeros for the dispersion relation for 
% the standard test case $k=0.5$
%
% 
disp(OMEGA(kgrid==0.3,1:3).')




% plot(OMEGA(3,:))

