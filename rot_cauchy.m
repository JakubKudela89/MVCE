function X=rot_cauchy(n,m,rnd,a);
% Generates a rotated Cauchy-distributed n x m matrix with scale a.
if nargin < 3, rng('default'); else, rng(rnd); end;
if nargin < 4, a = 1; end;
b=randn(m,1);
c=randn(m,1);
d=a*(b./c);
X = randn(n,m);
d = d ./ (sqrt(sum(X.^2,1)))';
X = X*spdiags(d,0,m,m);
return;
