function [u,R,factor,inds,iters] =...
solve_batch(y,tol,batch,KKY,maxit,print,u)
%This function implements the pooling and batching routines


if (nargin < 1), error('Please input X'); end
 [n,m] = size(y);
 if (nargin < 2), tol = 1e-07; end
 if (nargin < 3), batch = n; end
 if (nargin < 4), KKY = 0; end;
 if (nargin < 5), maxit = 100000; end;
 if (nargin < 6), print = 0; end;

inds = [1:n];
y2 = y(:,inds);
[u,R,factor] = minvol(y2,0.00000001,0,100000,0);
L = factor^(-1/2) * R;
Li = inv(L);
val1 = Li'*y;
res = sum(val1.*val1,1)-n;
[valk,indk] = maxk(res,batch);
iters = 1;
u_in = zeros(length(inds)+length(indk),1);
u_in(inds) = u;
inds = [inds,indk]; 

while 1
    y2 = y(:,inds);
    [u,R,factor] = minvol(y2,0.00000001,0,100000,0,u_in);
    L = factor^(-1/2) * R;
    Li = inv(L);
    val1 = Li'*y;
    res = sum(val1.*val1,1)-n;
    [valk,indk] = maxk(res,batch);
    indk = indk(valk > 0);
    iters = iters + 1;
    sd = setdiff(indk,inds);
    if isempty(sd)
        break;
    else
       u_in = zeros(length(inds)+length(indk),1);
       u_in(1:length(inds)) = u;
       inds = [inds,indk]; 
    end
end

u_real = zeros(m,1);
u_real(inds) = u;
u = u_real;

end

