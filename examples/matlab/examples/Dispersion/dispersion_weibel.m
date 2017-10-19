%% Dispersion relation for the Weibel Instability
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%
% Use MATLABs symbolic toolbox, and define
% frequency $\omega$, wave vector $k$
syms omega k x xi

% Set the number of significant decimal digits
% for variable-precision arithmetic
digits(16); 

% Set up grid of wave vectors the dispersion relation should be solved for
kgrid=0.5:0.05:1.5;
% kgrid=1.25;
% k=1.25;
% linspace(0.1,,25+1);
numz=1; %Number of tries to find a zero


%% Plasma dispersion function Z(x)
% The plasma dispersion function can be found in many textbooks
Z=symfun( sqrt(sym(pi))*exp(-x^2)*(1j-erfi(x)),x);

Phi=symfun( exp(-x^2/2)*int(exp(0.5*xi^2),xi,-1j*inf,x),x);



% Define parameters
vth=1;    %Thermal velocity of electrons
omegap=1; %Plasma frequency
sigma1=0.02/sqrt(2);
sigma2=sqrt(12)*sigma1;
%% Dispersion relation

D=symfun( omega.^2-k^2 + (sigma2/sigma1).^2 - 1 - ...
    (sigma2/sigma1)^2*Phi(omega/sigma1/k)*omega/sigma1/k,...
    [omega,k]);
% D=symfun(1+ (omegap/vth/k)^2*(1+ omega/vth/sqrt(2)/k*...
%     Z(omega/vth/sqrt(2)/k)), ...
%     [omega,k]);

% $$f(t=0,x,v)=\frac{1}{\sqrt{2 \pi}} 
%  e^{- \frac{1}{2} \left( \frac{v}{v_{th}}\right)^2}$$
%




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

%% Visualize result
figure;
plot(kgrid,abs(real(OM)),'*'); grid on;
xlabel('k'); ylabel('frequency');
title('zeros of dispersion relation')

figure;
plot(kgrid,-imag(OM),'*'); grid on;
xlabel('k'); ylabel('damping rate');
title('zeros of dispersion relation')

%% Standard test case $k=1.25$
% Found the following zeros for the dispersion relation for 
% the standard test case $k=0.5$
%
format long
disp(OM(kgrid==1.25,:).')

% 
% OM(kgrid==1.25,:)/2









