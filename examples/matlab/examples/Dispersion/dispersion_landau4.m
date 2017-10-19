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
kgrid=0.01:0.02:0.6;

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
%zk=vpa(exp(1j*2*sym('pi')*(1:n)/n));
zk=exp(sym('1j*2*pi')*(1:n)/n);

OMEGA=zeros(length(kgrid),K); % Roots of the Dispersion relation
RES=zeros(length(kgrid),K);   % Residual for Postprocessing
% Set the default residual to infinity in order to distinguish non zeros
RES(:,:)=Inf;

for idx=1:length(kgrid)
    
    % Print status
    fprintf('k= %08.4f,  %05.2f%%\n',...
        kgrid(idx), (idx-1)/length(kgrid)*100);
    
    % Alternative: use the ratdisk_K method
    % gk=double(vpa(1./D(zk*radius,kgrid(idx))));
    % [r,a,b,mu,nu,w] = ratdisk(gk, n-1-K, K);
    
    %% Compute Zeroes and Poles by Cauchy Integrals
    % Algorithm [Cz2] taken from
    %
    % NUMERICAL ALGORITHMS BASED ON ANALYTIC FUNCTION VALUES AT ROOTS OF
    % UNITY,
    % ANTHONY P. AUSTIN , PETER KRAVANJA , AND LLOYD N. TREFETHEN,
    % SIAM J. NUMER. ANAL. Vol. 52, No. 4, pp. 1795–1821,
    %
    s = ifft(double(1./D( zk*radius, kgrid(idx)  )));
    H = hankel(s(2:K+1), s(K+1:2*K));
    H2 = hankel(s(3:K+2), s(K+2:2*K+1));
    w = eig(H2,H);
    
    w=w*radius;
    
    %%
    % Evaluate residual
    % $|D(\omega,k)|$
    %
    res=double(abs(D(w,kgrid(idx))));
    
    % Sort by residual
    [res,I]=sort(abs(res));
    w=w(I);
    
    RES(idx,1:length(res))=res(:);
    OMEGA(idx,1:length(w))=w(:);
    
    
end

% Roots outside of the domain
OMEGA(abs(OMEGA)>radius)=0;
RES(OMEGA==0)=Inf;


%% Visualize result
% The roots $\omega$ are approximations, but sorted according to the
% residual for every $k$. We take the first _Nz_ roots.
%
%
Nz=2;
w=OMEGA(:,1:Nz);

figure;
plot(kgrid,real(w),'*'); grid on;
xlabel('k'); ylabel('frequency');
title('zeros of dispersion relation')

figure;
plot(kgrid,-imag(w),'*'); grid on;
xlabel('k'); ylabel('damping rate');
title('zeros of dispersion relation')
drawnow;


%% Standard test case $k=0.5$
% Found the following zeros for the dispersion relation for
% the standard test case $k=0.5$
%
%
disp(OMEGA(kgrid==0.5,1:3).')




%% Newton Method

maxit=15;  % Maximum number of iterations
tol=1e-3; % Tolerance

% Define Matlab Function for double precision Newton method 
% from symbolic functions
newton_step=matlabFunction...
    (simplify(D/diff(D,omega)),'Vars',[omega,k]);
residual=matlabFunction(abs(D),'Vars',[omega,k]);

tic;
test2=double(abs(D(rand(1,1),0.3)));
toc;
tic;
test1=residual(rand(1,1),0.3);
toc;

%Structure containing found roots in loose order
roots=struct('omega',[],'k',[],'residual',[],'iterations',[]);
num_roots=0;

for kdx=1:length(kgrid)
    fprintf('k= %08.4f,  %05.2f%%\n',...
        kgrid(kdx), (kdx-1)/length(kgrid)*100);
    for odx=1:size(OMEGA,2)
        w=OMEGA(kdx,odx);
        res=abs(RES(kdx,odx));
        
        if ( (w~=0) && ~isinf(w) && ~isnan(w) && ~isinf(res))
            it=0;
            while (res>tol && it < maxit)
                it=it+1;
                w=w-newton_step(w, kgrid(kdx));
                res=residual(w, kgrid(kdx));
            end
            
            % Check if valid root and add to list
            if (res<tol && w~=0 && ~isnan(res))
              num_roots=num_roots+1;
                roots(num_roots)=...
                struct('omega',w,...
                        'k',kgrid(kdx),...
                        'residual',res,...
                        'iterations',it);
                    
            end
        end
    end
end
%%
fprintf('Found %d roots', num_roots);


figure;
 plot( [roots.k] ,real([roots.omega]),'*'); grid on;
 xlabel('k'); ylabel('frequency');
 title('zeros of dispersion relation')

  figure;
 plot([roots.k] ,-imag([roots.omega]),'*'); grid on;
 xlabel('k'); ylabel('damping rate');
 title('zeros of dispersion relation')

 
 
 w=[roots.omega];
 k=[roots.k];
 
 
 for idx=1:10
     tic;
       w=w-newton_step(w, k);   
      toc;
 end
 
 res=abs(residual(w,k));
 valid=(res<1e3) & abs(w)<radius;
 
 

    
 figure;
 plot( [roots(valid).k] ,real([roots(valid).omega]),'*'); grid on;
 xlabel('k'); ylabel('frequency');
 title('zeros of dispersion relation')

  figure;
 plot([roots(valid).k] ,-imag([roots(valid).omega]),'*'); grid on;
 xlabel('k'); ylabel('damping rate');
 title('zeros of dispersion relation')


    
    
    
%Structure containing found roots in loose order
roots=struct('omega',[],'k',[],'residual',[],'iterations',[]);
num_roots=0;

for kdx=1:length(kgrid)
    for odx=1:size(OMEGA,2)
        w=OMEGA(kdx,odx);
        res=abs(RES(kdx,odx));
        if ( (w~=0) && ~isinf(w) && ~isnan(w) && ~isinf(res) && abs(w)<radius)
              num_roots=num_roots+1;
                roots(num_roots)=...
                struct('omega',w,...
                        'k',kgrid(kdx),...
                        'residual',res,...
                        'iterations',[]);
        end
        
    end
end