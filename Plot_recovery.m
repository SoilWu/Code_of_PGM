%% **************************************************************
%  filename: Plot_recovery
%% ***************************************************************
%% to plot the sucessful recovery of PGM and GPower_l0 
%% 
%% Copyright by Shaohua Pan and Yuqia Wu, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"

addpath('D:\test_SPCA\solvers')
 
addpath('D:\test_SPCA\data')

%% ****************** to generate the data **********************

nexample = 500;

dim = 500;

ns = [50:25:540];

m = length(ns);

rho = 1.0e-1;

kappa = 10;

mu = zeros(1,dim);

d = [400;300;ones(dim-2,1)];

D = diag(d);

w = 1/sqrt(10)*ones(kappa,1);

%% *********** parameters for PGM with extrapolation **************

OPTIONS_PGD.tol = 1.0e-6;

OPTIONS_PGD.printyes = 0;

OPTIONS_PGD.maxiter = 3000;

%% **************************************************************

PGD_result1 = zeros(nexample,m);

PGD_result2 = zeros(nexample,m);

GPM_result1 = zeros(nexample,m);

GPM_result2 = zeros(nexample,m);

PGD_sratio1 = zeros(m,1);

GPM_sratio1 = zeros(m,1);

PGD_sratio2 = zeros(m,1);

GPM_sratio2 = zeros(m,1);

for j=1:m
    
    for i = 1:nexample
        
        randstate = (j-1)*nexample+i
        
        randn('state',double(randstate));
        
        rand('state',double(randstate));
        
        temp_Sigma = randn(dim,dim);
        
        temp_Sigma = (temp_Sigma + temp_Sigma')/2;  % do not forget the symmetrization !!!!
        
        [P,~] = eig(temp_Sigma);
        
        P(:,1:2) = zeros(dim,2);
        
        P(1:kappa,1) = w;
        
        P(kappa+1:2*kappa,2) = w;
        
        v1 = P(:,1);
        
        v2 = P(:,2);
        
        Sigma = P*D*P';
        
        X = mvnrnd(mu,Sigma,ns(j));
        
        A = X'*X;
        
        %% ***************** to estimate ||A|| ***************************
        %%
        options.tol = 1e-6;
        options.issym = 1;
        options.disp  = 0;
        options.v0 = randn(dim,1);
        [xint,Asnorm] =eigs(@(y)(A*y),dim,1,'LM',options);
        
        lambda = rho*Asnorm;
        
        OPTIONS_PGD.Lipconst = 2.0001*Asnorm;
        
        xopt1 = PGD_L0sphere(xint,-A,OPTIONS_PGD,lambda,0);
        
        PGD_result1(j,i) = abs(xopt1'*v1);
        
        Axopt1 = A*xopt1;
        
        Axopt1_xopt1 = Axopt1*xopt1';
        
        A1 = A - (Axopt1_xopt1+Axopt1_xopt1') + (Axopt1'*xopt1)*(xopt1*xopt1');
        
        %% ***************** to estimate ||A|| ***************************
        %%
        options.tol = 1e-6;
        options.issym = 1;
        options.disp  = 0;
        options.v0 = randn(dim,1);
        [xint,Asnorm] =eigs(@(y)(A1*y),dim,1,'LM',options);
        
        OPTIONS_PGD.Lipconst = 2.0001*Asnorm;
        
        lambda = rho*Asnorm;
        
        xopt2 = PGD_L0sphere(xint,-A1,OPTIONS_PGD,lambda,0);
        
        PGD_result2(j,i) = abs(xopt2'*v2);
        
      %% ************* Generalized power method ********************
        
        yopt1 = GPower(A,1/4,1,'l0',0);
        
        GPM_result1(j,i) = abs(yopt1'*v1);
        
        Ayopt1 = A*yopt1;
        
        Ayopt1_yopt1 = Ayopt1*yopt1';
        
        A1 = A-(Ayopt1_yopt1+Ayopt1_yopt1') +(Ayopt1'*yopt1)*(yopt1*yopt1');
        
        yopt2 = GPower(A1,1/4,1,'l0',0);
        
        GPM_result2(j,i) = abs(yopt2'*v2);
        
    end
    
    PGD_sratio1(j) = length(find(PGD_result1(j,:)>0.99))/nexample;
    
    PGD_sratio2(j) = length(find(PGD_result2(j,:)>0.99))/nexample;
    
    GPM_sratio1(j) = length(find(GPM_result1(j,:)>0.99))/nexample;
    
    GPM_sratio2(j) = length(find(GPM_result2(j,:)>0.99))/nexample;
end

save('recover_result','PGD_sratio1','PGD_sratio2','GPM_sratio1','GPM_sratio2');

subplot(1,2,1);
h1=plot(ns,PGD_sratio1,'r-o',ns, GPM_sratio1, 'b-');  
set(h1,'LineWidth',2.5) 
xlabel('(a) Size of sample');   ylabel('Recoverability');
set(get(gca,'XLabel'),'FontSize',10);
set(get(gca,'YLabel'),'FontSize',10);
legend('PGD','GPower\_l0');
hold on;
subplot(1,2,2);
h2=plot(ns,PGD_sratio2,'r-o',ns, GPM_sratio2, 'b-');  
set(h2,'LineWidth',2.5) 
xlabel('(b) Size of sample');   ylabel('Recoverability');
set(get(gca,'XLabel'),'FontSize',10);
set(get(gca,'YLabel'),'FontSize',10);
legend('PGD','GPower\_l0');
hold on;
