clear;clc;close all;
% The files implement the wolfe-antwood algorithm modified by pooling and batching:
% The paper can be found here: https://doi.org/10.13164/mendel.2019.2.019
% created by: Jakub Kudela (Jakub.Kudela@vutbr.cz)

dim = 20;               % problem dimension
n_points = 100000;      % number of points

% cauhy distribution
% X=rot_cauchy(dim,n_points);

% normal distribution
W = randn(dim); X = randn(n_points, dim); X=X*W; X=X';

tol = 1e-7;

tic
    [u1,R1,factor1] = minvol(X,tol,0,100000,0); % solution without batching
t1 = toc

batch = 2000;       % size of the batch
tic
    [u2,R2,factor2,inds2,iters2] = solve_batch(X,tol,batch); % solution with batching
t2 = toc

% comparing the resulting elipses: 
% L1 and L2 should be close (apart from signs)
% res1, res2 are the "x_i^T*H*x_i - n " values for each point for the two elipses
L1 = factor1^(-1/2) * R1;
Li1 = inv(L1);
val1 = X'*Li1;
res1 = sum(val1.*val1,2)-dim;
L2 = factor2^(-1/2) * R2;
Li2 = inv(L2);
val2 = X'*Li2;
res2 = sum(val2.*val2,2)-dim;
max(abs(res1-res2))
