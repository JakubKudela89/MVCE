 function var = updatevar(var,lam,mult,Mxj,X)

% Update the vector var of variances after a rank-one change

 tmp = Mxj' * X;
 var = (1 + lam) * (var - mult * (tmp.^2));
 return;