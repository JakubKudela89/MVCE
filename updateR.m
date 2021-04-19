function [R,factor,down_err] = updateR(R,factor,xj,lam);

% updates the Cholesky factor R

p=0;
xx = sqrt(abs(lam)*factor) * xj;
if (lam > 0), R = cholupdate(R,xx,'+'); 
   else, [R,p] = cholupdate(R,xx,'-'); 
end;
factor = factor * (1 + lam);
if (p>0), down_err=1; else down_err=0; end;
return;