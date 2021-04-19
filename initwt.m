 function u = initwt(X,print)

%  obtains the initial weights u using the Kumar-Yildirim algorithm,
%  taking into account that X represents [X,-X].

 if (nargin < 2), print = 0; end;
 if print, st = cputime; end;
 [n,m] = size(X);
 u = zeros(m,1);
 Q = eye(n);
 d = Q(:,1);

% Q is an orthogonal matrix whose first j columns span the same space
% as the first j points chosen X(:,ind) (= (X(:,ind) - (-X(:,ind)))/2).

 for j = 1:n,

%     compute the maximizer of | d'*x | over the columns of X.

     dX = abs(d'*X);
     [maxdX,ind] = max(dX);
     u(ind) = 1;
     if j == n, break, end;

%     update Q.

     y = X(:,ind);
     z = Q'*y;
     if j > 1, z(1:j-1) = zeros(j-1,1); end;
     zeta = norm(z); zj = z(j); if zj < 0, zeta = - zeta; end;
     zj = zj + zeta; z(j) = zj;
     Q = Q - (Q * z) * ((1/(zeta*zj)) * z');
     d = Q(:,j+1);
 end;
 u = u / n;
 if print, 
    fprintf('\n Initialization time = %5.2f \n',cputime - st);
 end;
 return;