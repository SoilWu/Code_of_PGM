%% **************************************************************
%  filename: Plot_covariance
%% ***************************************************************
%% to plot the proportion of explained covariance for two solvers
%% PGM and GPower_l0
%%
%% Copyright by Shaohua Pan and Yuqia Wu, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"

addpath('D:\test_SPCA\solvers')
 
addpath('D:\test_SPCA\data')

%% ******************* to upload the data ***********************

X = xlsread('leukemia.xlsx');

[m,n] = size(X);

Xt = X';

X = X - mean(Xt)'*ones(1,n);  % to such that each row of X has a zero mean

A = X*X';

%%
%% ***************** to estimate ||A|| ***************************
%%
options.tol = 1e-6;
options.issym = 1;
options.disp  = 0;
options.v0 = randn(m,1);
[x0,Asnorm] =eigs(@(y)(A*y),m,1,'LM',options);

variance = Asnorm;

%% *********** parameters for PGM with extrapolation **************

OPTIONS_PGD.tol = 1.0e-6;

OPTIONS_PGD.printyes = 0;

OPTIONS_PGD.Lipconst = 2.0001*Asnorm;

OPTIONS_PGD.maxiter = 3000;

gamma = 0;    % or 1.0e-3*Asnorm

P = linspace(0.0000001,0.0006,50);

Q = linspace(0.000005, 0.02, 50);

x1 = zeros(50,1);

y1 = zeros(50,1);

x2 = zeros(50,1);

y2 = zeros(50,1);

time1 = zeros(50,1);

time2 = zeros(50,1);

for i = 1:50  
    
    lambda = P(i)*Asnorm; 

    [xopt,loss,time1(i)] = PGD_L0sphere(x0,-A,OPTIONS_PGD,lambda,gamma);
    
    svariance = abs(loss);
     
    x1(i) = sum(abs(xopt)>1.0e-8*abs(xopt));

    y1(i) = svariance/variance; 
end
    
for j = 1:50  
    
    tstart = clock;
    
    xopt = GPower(A,Q(j),1,'l0',0);
    
    time2(j) = etime(clock,tstart);
    
    x2(j) = sum(abs(xopt)>1.0e-8*abs(xopt));
    
    y2(j) = xopt'*(A*xopt)/variance;
 
end

subplot(2,1,1);
plot(x1,y1,'r-o',x2,y2,'b--','LineWidth',2);
xlabel('Cardinality');
ylabel('Proportion of explained variance(%)');
legend('PGD','GPower\_l0')
hold on;
% 
subplot(2,1,2);
plot(x1,time1,'r-o',x2,time2,'b-o','LineWidth',2);
xlabel('Cardinality');
ylabel('Computing time(s)');
legend('PGD','GPower\_l0')
