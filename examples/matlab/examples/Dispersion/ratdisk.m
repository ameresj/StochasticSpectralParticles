function [r,a,b,mu,nu,poles,residues] = ratdisk(f,m,n,N,tol)
%  Input: Function f or vector of data at zj = exp(2i*pi*(0:N)/(N+1))
%           for some N>=m+n.  If N>>m+n, it is best to choose N odd.
%         Maximal numerator, denominator degrees m,n.
%         An optional 5th argument specifies relative tolerance tol.
%           If omitted, tol = 1e-14.  Use tol=0 to turn off robustness.
% Output: function handle r of exact type (mu,nu) approximant to f
%         with coeff vectors a and b and optional poles and residues.
% P. Gonnet, R. Pachon, L. N. Trefethen, January 2011

if nargin<4, if isfloat(f), N=length(f)-1;
    else N=m+n; end, end                       % do interpolation if no N given
N1 = N+1;                                    % no. of roots of unity
if nargin<5, tol = 1e-14; end                % default rel tolerance 1e-14
if isfloat(f), fj = f(:);                    % allow for either function
else fj = f(exp(2i*pi*(0:N)'/(N1))); end     %   handle or data vector
ts = tol*norm(fj,inf);                       % absolute tolerance
M = floor(N/2);                              % no. of pts in upper half-plane
f1 = fj(2:M+1); f2 = fj(N+2-M:N1);           % fj in upper, lower half-plane
realf = norm(f1(M:-1:1)-conj(f2),inf)<ts;    % true if fj is real symmetric
oddN = mod(N,2)==1;                          % true if N is odd
evenf = oddN & norm(f1-f2,inf)<ts;           % true if fj is even
oddf  = oddN & norm(f1+f2,inf)<ts;           % true if fj is odd
row = conj(fft(conj(fj)))/N1;                % 1st row of Toeplitz matrix
col = fft(fj)/N1; col(1) = row(1);           % 1st column of Toeplitz matrix
if realf, row = real(row);                   % discard negligible imag parts
    col = real(col); end
d = xor(evenf,mod(m,2)==1);                  % either 0 or 1
while true                                   % main stabilization loop
    Z = toeplitz(col,row(1:n+1));              % Toeplitz matrix
    if ~oddf && ~evenf                          % fj is neither even nor odd
        [~,S,V] = svd(Z(m+2:N1,:),0);            % singular value decomposition
        b = V(:,n+1);                            % coeffs of q
    else                                       % fj is even or odd
        [~,S,V] = svd(Z(m+2+d:2:N1,1:2:n+1),0);  % special treatment for symmetry
        b = zeros(n+1,1); b(1:2:end) = V(:,end); % coeffs of q
    end
    if N > m+n && n>0, ssv = S(end,end);       % smallest singular value
    else ssv = 0; end                          % or 0 in case of interpolation
    qj = ifft(b,N1); qj = qj(:);               % values of q at zj
    ah = fft(qj.*fj);                          % coeffs of p-hat
    a = ah(1:m+1);                             % coeffs of p
    if realf a = real(a); end                  % discard imag. rounding errors
    if evenf a(2:2:end) = 0; end               % enforce even symmetry of coeffs
    if  oddf a(1:2:end) = 0; end               % enforce odd symmetry of coeffs
    if tol>0                                   % tol=0 means no stabilization
        ns = n;                                  % no. of singular values
        if oddf||evenf, ns = floor(n/2); end
        s = diag(S(1:ns,1:ns));                  % extract singular values
        nz = sum(s-ssv<=ts);                     % no. of sing. values to discard
        if nz == 0, break                        % if no discards, we are done
        else n=n-nz; end
    else break                                 % no iteration if tol=0.
    end
end                                          % end of main loop
nna = abs(a)>ts; nnb = abs(b)>tol;           % nonnegligible a and b coeffs
kk = 1:min(m+1,n+1);                         % indices a and b have in common
a = a(1:find(nna,1,'last'));                 % discard trailing zeros of a
b = b(1:find(nnb,1,'last'));                 % discard trailing zeros of b
if isempty(a) % special case of zero function
    a=0; b=1;
end
mu = length(a)-1; nu = length(b)-1;          % exact numer, denom degrees
r = @(z) polyval(a(end:-1:1),z)...           % function handle for r
    ./polyval(b(end:-1:1),z);
if nargout>5                                 % only compute poles if necessary
    poles = roots(b(end:-1:1));                % poles
    t = max(tol,1e-7);                         % perturbation for residue estimate
    residues = t*(r(poles+t)-r(poles-t))/2;    % estimate of residues
end
