%% *************************************************************
% filename: Table_Pitprops
%% *************************************************************
%% 
%% Copyright by Shaohua Pan and Yuqia Wu, 2018/11/8
%  "KL property of exponent 1/2 of the zero-norm regularized 
%  quadratic function on sphere and application"
%%

addpath('D:\Wu_Yuqia\test_SPCA\solvers')

addpath('D:\Wu_Yuqia\test_SPCA\data')

randstate = 110;
randn('state',double(randstate));
rand('state',double(randstate));

A = ([[1, 0.954, 0.364, 0.342, -0.129, 0.313, 0.496, 0.424, 0.592, 0.545, 0.084, -0.019, 0.134],
         [0.954, 1, 0.297, 0.284, -0.118, 0.291, 0.503, 0.419, 0.648, 0.569, 0.076, -0.036, 0.144],
         [0.364, 0.297, 1, 0.882, -0.148, 0.153, -0.029, -0.054, 0.125, -0.081, 0.162, 0.220, 0.126],
         [0.342, 0.284, 0.882, 1, 0.220, 0.381, 0.174, -0.059, 0.137, -0.014, 0.097, 0.169, 0.015],
         [-0.129, -0.118, -0.148, 0.220, 1, 0.364, 0.296, 0.004, -0.039, 0.037, -0.091, -0.145, -0.208],
         [0.313, 0.291, 0.153, 0.381, 0.364, 1, 0.813, 0.090, 0.211, 0.274, -0.036, 0.024, -0.329],
         [0.496, 0.503,-0.029, 0.174, 0.296, 0.813, 1, 0.372, 0.465, 0.679, -0.113, -0.232, -0.424],
         [0.424, 0.419, -0.054, -0.059, 0.004, 0.090, 0.372, 1, 0.482, 0.557, 0.061, -0.357, -0.202],
         [0.592, 0.648, 0.125, 0.137, -0.039, 0.211, 0.465, 0.482, 1,  0.526, 0.085, -0.127, -0.076],
         [0.545, 0.569, -0.081, -0.014, 0.037, 0.274, 0.679, 0.557, 0.526, 1, -0.319, -0.368, -0.291],
         [0.084, 0.076, 0.162, 0.097, -0.091, -0.036, -0.113, 0.061, 0.085, -0.319, 1, 0.029, 0.007],
         [-0.019, -0.036, 0.220, 0.169, -0.145, 0.024, -0.232, -0.357, -0.127, -0.368, 0.029, 1, 0.184],
         [0.134, 0.144, 0.126,  0.015, -0.208, -0.329, -0.424, -0.202, -0.076, -0.291, 0.007, 0.184, 1]]);

if (norm(A-A','fro')>1.0e-12)
    display('A is not a covariance matrix')
    return;
end

n = size(A,1);

[P,D] = eig(A);

X = P*diag(diag(D).^(1/2))*P';

k = 6;       % the number of PCs 

rho = 0.1;

rho1 = 0.15;

%% *********** parameters for PGM with extrapolation **************

OPTIONS_PGD.tol = 1.0e-6;

OPTIONS_PGD.printyes = 0;

OPTIONS_PGD.maxiter = 3000;

%% **************************************************************

PGD_eigvec = zeros(n,k);

PGD_nzeigv = zeros(k,1);

PGD_variance = zeros(k,1);

PGD_IVar = zeros(k,1);

GPM_eigvec = zeros(n,k);

GPM_nzeigv = zeros(k,1);

GPM_variance = zeros(k,1);

GPM_IVar = zeros(k,1);

xopt = zeros(n,k);

yopt = zeros(n,k);

%% ******************* The PGD Method *****************************

tempA = A;

for j = 1:k
    
    %% ***************** to estimate ||tempA|| ***************************
    %%
    options.tol = 1e-6;
    options.issym = 1;
    options.disp = 0;
    options.v0 = randn(n,1);
    
    [xint,Asnorm] = eigs(@(y)(tempA*y),n,1,'LM',options);

    OPTIONS_PGD.Lipconst = 2.0001*Asnorm;
    
    lambda = rho*Asnorm;
    
    tempA_old = tempA;
    
    [xoptj,loss] = PGD_L0sphere(xint,-tempA_old,OPTIONS_PGD,lambda,0);
    
    xopt(:,j) = xoptj;
  
    abs_xoptj = abs(xoptj);
    
    PGD_variance(j) = abs(loss);
    
    PGD_eigvec(:,j) = xoptj;
    
    PGD_nzeigv(j) = sum(abs_xoptj>1.0e-8*max(abs_xoptj));
    
    Axopt = tempA*xoptj;

    Axopt_xopt = Axopt*xoptj';

    tempA = tempA_old - (Axopt_xopt + Axopt_xopt') + PGD_variance(j)*(xoptj*xoptj');

end

PGD_eigvec

Variance = PGD_variance/13

Zx = X*xopt;

[Qx, Rx] = qr(Zx);

rx = diag(Rx);

for i = 1:k
    
   PGD_IVar(i) = sum(norm(rx(1:i)).^2)/13;
   
end

adj_PGD_variance = PGD_IVar(2:end) - PGD_IVar(1:end-1)

CA_PGD_variance = PGD_IVar

%% *********************** GPower Method **************************

tempA = A;

for j = 1:k
    
    yoptj = GPower(tempA,rho1,1,'l0',0);
        
    yopt(:,j) = yoptj;
    
    abs_yoptj = abs(yoptj);
    
    GPM_eigvec(:,j) = yoptj;
    
    GPM_nzeigv(j) = sum(abs_yoptj >1.0e-8*max(abs_yoptj));
    
    Ayopt = tempA*yoptj;
    
    GPM_variance(j)= yoptj'*Ayopt;
    
    %% ******************* The second one *****************************
    
    Ayopt_yopt = Ayopt*yoptj';
    
    tempA = tempA-(Ayopt_yopt + Ayopt_yopt') + GPM_variance(j)*(yoptj*yoptj');
end


GPM_eigvec

Variance = GPM_variance/13
 
Zy = X*yopt;
[Qy,Ry] = qr(Zy);

ry = diag(Ry);

for i=1:k
    
   GPM_IVar(i) = sum(norm(ry(1:i)).^2)/13;
end

adj_GPM_variance = GPM_IVar(2:end) - GPM_IVar(1:end-1)

CA_GPM_variance = GPM_IVar
