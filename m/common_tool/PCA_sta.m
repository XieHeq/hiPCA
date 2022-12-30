function [T2_1, Q_1, F_1,T2lim,SPElim,Flim,Cw,D,fai] = PCA_sta(x_1,P,NumVari,PCs,lamda,confr)
% x_1= X_normal_valid;%   Testing data, fault operation
% x_1= X_fault_valid;
% x_1= X_fault_test;
[NumSampTest,NumVariTest]= size(x_1);
if NumVariTest ~= NumVari%   check the dimension of the testing data
    error('Test Data Dimension Mismatch!')
end
% Score_Train = x*P;
Score_Test= x_1*P;%   scores of testing data
x_rec_1= Score_Test*P';%   reconstructed data 
e1= (x_1-x_rec_1).^2;
Q_1= sum(e1,2);%   Q statistics
T2_1= zeros(NumSampTest,1);%   T^2 statistics
for i= 1:NumSampTest
    for j= 1:PCs
        T2_1(i,1)= T2_1(i,1)+Score_Test(i,j)*Score_Test(i,j)/lamda(j);
    end
end
T2lim = chi2inv(confr,PCs);
% T2lim= PCs*(NumSamp-1)*(NumSamp+1)/NumSamp/(NumSamp-PCs)*finv(0.99,PCs,NumSamp);
theta_1= 0;
theta_2= 0;
theta_3= 0;
for i= PCs+1:NumVari
    theta_1= theta_1+lamda(i);
    theta_2= theta_2+lamda(i)^2;
    theta_3= theta_3+lamda(i)^3;
end
g_spe = theta_2/theta_1;
h_spe = theta_1^2/theta_2;
SPElim = g_spe*chi2inv(confr,h_spe);

% h0= 1-2*theta_1*theta_3/(3*theta_2^2);
% SPElim= theta_1*(h0*norminv(0.99,0,1)*sqrt(2*theta_2)/theta_1+1+theta_2*h0*(h0-1)/theta_1^2)^(1/h0);
Cw=eye(NumVari)-P*P';
D=P*diag(lamda(1:PCs).^-1)*P';
fai = Cw/SPElim+D/T2lim;
% F_1 = diag(x_1*fai*x_1');%   Combined statistics
F_1 = Q_1/SPElim + T2_1/T2lim;
g_Fi = (PCs/T2lim/T2lim+theta_2/SPElim/SPElim)/(PCs/T2lim+theta_1/SPElim);
h_Fi = (PCs/T2lim+theta_1/SPElim)^2/(PCs/T2lim/T2lim+theta_2/SPElim/SPElim);
Flim = g_Fi*chi2inv(confr,h_Fi);
end