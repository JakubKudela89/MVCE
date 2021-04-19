 function [u,R,factor,improv,mxv,mnv,flagstep,lamhist,var,time,iter,act] =...
     minvol(X,tol,KKY,maxit,print,u)

% Finds the minimum-volume ellipsoid containing the columns of X using the
% Fedorov-Wynn-Frank-Wolfe method, with Wolfe-Atwood away steps if KKY = 0.
% The algorithm also uses the method of Harman and Pronzato to
% eliminate points that are found to be inessential.
%
% The algorithm returns an ellipsoid providing a (1+tol)n-rounding of
% the convex hull of the columns of X in n-space. Set tol to eps/n to get
% a (1+eps)-approximation of the minimum-volume ellipsoid.
%
%%%%%%%%%%%%%%%%%%%%  INPUT PARAMETERS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% X is the input data set.
%
% tol is the tolerance (measure of duality gap), set to 10^-7 by default;
%
% KKY is:
%     0 (default) for the Wolfe-Atwood method using Wolfe's away steps
%     (sometimes decreasing the weights) with the Kumar-Yildirim start;
%     1 if using the Fedorov-Wynn-Frank-Wolfe algorithm
%     (just increasing the weights) with the Khachiyan initialization;
%     2 for the Wolfe-Atwood method with the Khachiyan initialization;
%
% maxit is the maximum number of iterations (default 100,000);
%
% print is the desired level of printing (default 1); and
%
% u is the initial value for the weights (default as above).
%
%%%%%%%%%%%%%%%%%%%%%  OUTPUT PARAMETERS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    u determines the optimal weights on the m columns of X (U = Diag(u));
%
%    R is a scaled (upper triangular) Cholesky factor of 
%       M := XUX': R^T*R = factor * XUX';
%
%    improv(i) gives the objective improvement at iteration i;
%
%    mxv(i) gives the maximum variance (x_i^T M^{-1} x_i) at iteration i;
%
%    mnv(i) gives the minimum variance for those i with u_i positive 
%       at iteration i;
%
%    flagstep(i) identifies the type of step taken at iteration i: 1(drop),
%       2(decrease), 3(add), and 4(increase);
%
%    lamhist(i) holds the step length lam at iteration i;
%
%    var gives the variances of all the points at completion 
%       (points that have been eliminated are assigned the value -1);
%
%    iter is the total number of iterations taken;
%
%    act is the index set of active columns of X, those that have not 
%       been eliminated; and
%
%    time is the total cputime spent in order to obtain the optimal solution.
%
%    Calls initwt, updateR, updatevar, and ellipse if n = 2 to draw
%       the ellipse at each iteration.

%%%%%%%%%%%%%%%%%  INITIALIZE INPUT PARAMETERS IF NOT DEFINED  %%%%%%%%%%%%

 if (nargin < 1), error('Please input X'); end
 [n,m] = size(X);
 if (nargin < 2), tol = 1e-07; end
 if (nargin < 3), KKY = 0; end;
 if (nargin < 4), maxit = 100000; end;
 if (nargin < 5), print = 1; end;
 if print,
    fprintf('\n Dimension = %5.0f, Number of points = %5.0f',n,m)
    fprintf(', Tolerance = %5.1e \n',tol);
 end;
 if (nargin < 6),
    if (KKY >= 1),
       u = (1/m) * ones(m,1);
       fprintf('\n Using Khachiyan initialization \n');
    else
        u = initwt(X,print);
    end;
 end;

%%%%%%%%%%%%%%%%%  INITIALIZE NECESSARY PARAMETERS  %%%%%%%%%%%%%%%%%%%%%%%

 st = cputime;
 iter = 1;
 n100 = max([n,100]);
 n50000 = max([n,50000]);
 tol2 = 1e-08;
 mxv = zeros(1,maxit);    % pre-allocate memory for output vectors
 mnv = zeros(1,maxit);
 flagstep = zeros(1,maxit);
 lamhist = zeros(1,maxit);
 mvarerrhist = zeros(1,maxit); improv = zeros(1,maxit);

%%%%%%%%%%%%%%%%%  INITIALIZE CHOLESKY FACTOR  %%%%%%%%%%%%%%%%%%%%%%%%%%%

 upos = find(u > 0);
 lupos = length(upos);
 A = spdiags(sqrt(u(upos)),0,lupos,lupos)*X(:,upos)';  % A'A = M := XUX'
 [Q,R] = qr(A,0);
 factor = 1;                                   % M = factor^-1 * R' * R

% Draw the current ellipse if n = 2.

 if (n == 2),
    pause on;
    clf;
    radii = .02*ones(1,m);
    ellipse(radii,radii,zeros(1,m),X(1,:),X(2,:),'k',100);
    hold on;
    C = 'r';
    M = X(:,upos)*spdiags(u(upos),0,lupos,lupos)*X(:,upos)';
    [V,D] = eig(M);
    phi = atan(V(2,1)/V(1,1));
    aa = sqrt(2*D(1,1)); bb = sqrt(2*D(2,2));
    ellipse(aa,bb,phi,0,0,C,100);
    pause;
 end;

%%%%%%%%%%%%%%%%%%%%%%%%  INITIALIZE VARIANCES %%%%%%%%%%%%%%%%%%%%%%%%%%%%

 RX = R' \ X; 			    % RX = R^{-T} X
 var = sum(RX .* RX,1);		% var(i) = x_i^T M^{-1} x_i

% maxvar is the maximum variance.

 [maxvar,maxj] = max(var);

%%%%%%%%%%%%%%%%%% TRY ELIMINATING POINTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% act lists the mm non-eliminated columns of X, 
% and XX is the corresponding submatrix.

 act = 1:1:m;
 XX = X;
 mm = m; oldmm = m;

% Use the Harman-Pronzato test to see if columns of X can be eliminated.

 ept = maxvar - n;
 tresh = n * (1 + ept/2 - (ept*(4+ept-4/n))^.5/2);
 e = find(var > tresh | u' > tol2);
 act = act(e);
 XX = XX(:,e);
 mm = length(e);

% If only n columns remain, recompute u and R.

 if mm == n,
     u = (1/n)*ones(n,1);
     upos = find(u > tol2);
     A = spdiags(sqrt(u),0,mm,mm) * XX';
     [Q,R] = qr(A,0);
     factor = 1;
     RX = R' \ XX;
     var = sum(RX .* RX,1);
 else
     var = var(e);
     u = u(e)/sum(u(e));
     upos = find(u > tol2);
 end;
 if print, 
          fprintf('\n At iteration %6.0f', iter-1);
          fprintf(', number of active points %5.0f \n',length(act));
 end;
 oldmm = mm;

%%%%%%%%%%%%%%%%%%%%  FIND "FURTHEST" AND "CLOSEST" POINTS %%%%%%%%%%%%%%%%

 [maxvar,maxj] = max(var);
 [minvar,ind] = min(var(upos)); minj = upos(ind); mnvup = minvar;

% minj has smallest variance among points with positive weight.

 mxv(iter) = maxvar; mnv(iter) = minvar;
 if KKY==1, fprintf('\n Using KKY'); mnvup = n; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%  START ITERATIONS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

 while ((maxvar > (1+tol)*n) || (mnvup < (1-tol)*n)) && (iter < maxit),

%%%%%%%%%%%%%%%   SELECT THE COMPONENT TO INCREASE OR DECREASE  %%%%%%%%%%%

    if maxvar + mnvup > 2*n,
       j = maxj;
       mvar = maxvar;
    else
       j = minj;
       mvar = mnvup;
    end;

%    Compute Mxj = M^{-1} x_j and recompute var(j).

    flag_recompute = 0;
    xj = XX(:,j);
    Rxj = R' \ xj;
    Mxj = factor * (R \ Rxj);
    mvarn = factor * (Rxj' * Rxj);
    mvarerror = abs(mvarn - mvar)/max([1,mvar]);
    mvarerrhist(iter) = mvarerror;
    if (mvarerror > tol2),
       flag_recompute = 1;
    end;
    mvar = mvarn;

%%%%%%    COMPUTE STEPSIZE LAM (MAY BE NEGATIVE), EPSILON, AND  %%%%%%%%%%%
%%%%%%    IMPROVEMENT IN LOGDET                                 %%%%%%%%%%%

    lam = (mvar - n) / ((n-1) * mvar);
    ep = (mvar/n - 1);
    uj = u(j);
    lam = max(lam,-uj);
    lamhist(iter) = lam;                        % record the step size taken
    if lam < -u(j) + tol2, flagstep(iter) = 1;       % drop step
       elseif lam < 0, flagstep(iter) = 2;           % decrease step
       elseif u(j) < tol2, flagstep(iter) = 3;       % add step
       else flagstep(iter) = 4;                      % increase step
    end

%    Update u and make sure it stays nonnegative.

    imp = log(1 + lam*mvar) - n * log(1 + lam);
    uold = u;
    u(j) = max(uj + lam,0); u = (1/(1 + lam)) * u;
    upos = find(u > tol2);
    if (print) && (iter > 1) && (iter-1 == floor((iter-1)/n100) * n100),

%       Print statistics.

       fprintf('\n At iteration %6.0f, maxvar %9.5f',iter-1,maxvar)
       fprintf(', minvar %9.5f',minvar)
    end;

%%%%%%%%%    UPDATE (OR RECOMPUTE) CHOLESKY FACTOR AND VAR   %%%%%%%%%%%%%%

    if (iter > 1) && ((iter-1 == floor((iter-1)/n50000) * n50000) ...
                      || (flag_recompute && print)),
       upos = find(uold > 0);
       lupos = length(upos);
       M = XX(:,upos) * spdiags(uold(upos),0,lupos,lupos) * XX(:,upos)';
       normdiff = norm(factor*M - R'*R) / (factor*norm(M));
       if (normdiff > tol2),
          flag_recompute = 1;
       end;
       if (flag_recompute && print)
            fprintf('\n Relative error in mvar = %8.1e', mvarerror);
            fprintf(' and in XUX'' = %8.1e; reinverting \n', normdiff);
        end;
    end;

    if flag_recompute,
       upos = find(u > 0);
       lupos = length(upos);
       A = spdiags(sqrt(u(upos)),0,lupos,lupos) * XX(:,upos)';
       [Q,R] = qr(A,0);
       factor = 1;
       RX = R' \ XX;
       var = sum(RX .* RX,1);
    else

        % Update factorizations.

       [R,factor,down_err] = updateR(R,factor,xj,lam);
       if down_err, fprintf('\n Error in downdating Cholesky'); break; end;
       mult = lam / (1 + lam*mvar);
       var = updatevar(var,lam,mult,Mxj,XX);
    end;

%    Update maxvar.

    [maxvar,maxj] = max(var);

%    Use the Harman-Pronzato test to see if 
%    further columns can be eliminated.

    if (iter > 1) && (iter-1 == floor((iter-1)/n100) * n100),
       ept = maxvar - n;
       tresh = n * (1 + ept/2 - (ept*(4+ept-4/n))^.5/2);
       e = find(var > tresh | u' > tol2);
       if length(e) < mm,
          act = act(e);
          XX = XX(:,e);
          mm = length(e);
          if mm == n
              u = (1/n)*ones(n,1);
              uold = u;
              upos = find(u > tol2);
              A = spdiags(sqrt(u),0,mm,mm) * XX';
              [Q,R] = qr(A,0);
              factor = 1;
              RX = R' \ XX;
              var = sum(RX .* RX,1);
              [maxvar,maxj] = max(var);
          else
              var = var(e);
              u = u(e)/sum(u(e));
              uold = uold(e)/sum(uold(e));
              upos = find(u > tol2);
              [maxvar,maxj] = max(var);
          end;
          if (print == 2) || (print && (mm < oldmm / 2)),
                fprintf('\n \n At iteration %6.0f',iter - 1);
                fprintf(', number of active points %5.0f \n',length(act));
          end;
          oldmm = mm;
       end;
    end;

%    Update minvar, iteration statistics.

    upos = find(u > 0);
    [minvar,ind] = min(var(upos)); minj = upos(ind); mnvup = minvar;
    iter = iter+1;
    improv(iter) = imp;
    mxv(iter) = maxvar;
    mnv(iter) = minvar;
    if KKY == 1, mnvup = n; end;

%    Draw the current ellipse if n = 2.

    if (n == 2),
        if (C == 'r'), C = 'b';
           elseif (C == 'b'), C = 'g';
           elseif (C == 'g'), C = 'r';
        end;
        M = XX*spdiags(u,0,mm,mm)*XX';
        [V,D] = eig(M);
        phi = atan(V(2,1)/V(1,1));
        aa = sqrt(2*D(1,1)); bb = sqrt(2*D(2,2));
        ellipse(aa,bb,phi,0,0,C,100);
        pause;
    end;
 end;

%%%%%%%%%%%%%%%% CALCULATE AND PRINT SOME OF THE OUTPUT VARIABLES %%%%%%%%%

% Put back eliminated entries.

 mxv = mxv(1:iter); mnv = mnv(1:iter);
 flagstep = flagstep(1:iter); improv = improv(1:iter);
 lamhist = lamhist(1:iter);
 uu = zeros(m,1); uu(act) = u; u = uu;
 varr = -ones(m,1); varr(act) = var; var = varr;
 iter = iter - 1;

 if print,
    for i=1:4, cases(i) = length(find(flagstep==i)); end
    fu = find(u > 1e-12);
    fprintf('\n \n maxvar - n = %4.3e', max(var) - n)
    fprintf(', n - minvar = %4.3e \n', n - min(var(fu)));
    fprintf('\n Drop, decrease, add, increase cases: %6.0f', cases(1));
    fprintf('%6.0f %6.0f %6.0f \n',cases(2),cases(3),cases(4)),
    fprintf('\n Number of positive weights = %7.0f \n', length(fu));
    fprintf('\n Number of iterations       = %7.0f \n', iter);
    fprintf('\n Time taken                 = %7.2f \n \n', cputime - st);
 end;
                
 return;