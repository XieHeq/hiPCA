X_normal = fea_h(:,1:num_f);
X_normal_valid =  fea_n(:,1:num_f);
X_abnormal_valid =  fea2(y_test==0,1:num_f);
X_abnormal_invalid =  fea2(y_test==1,1:num_f);

x = X_normal;
[NumSamp, NumVari] = size(x);
[U,S,V]= svd(x/sqrt(NumSamp-1));%   singular value decomposition, x/sqrt(n-1)=USV'
sigma= zeros(NumVari,1);
for i= 1:NumVari
    sigma(i,1)= S(i,i);%   singular value
end
lamda= sigma.^2;%   characteristic value
percent_explained= lamda/sum(lamda);%   contribution
sum_per= 0;
for i= 1:NumVari
    sum_per= sum_per+percent_explained(i,1);
    if sum_per>= sum_pcs%   the first PCs largest characteristic values with accumulated contribution larger than 80%
        PCs= i;%   number of PC
        break;
    end
end
P= V(:,[1:PCs]);%   loading matrix
Pr = V(:,[PCs+1:end]);
%%
x_0= X_normal;%   Testing data, normal operation
[T2_0, Q_0, F_0,T2lim,SPElim,Flim,Cw0,D0,fai0] = PCA_sta(x_0,P,NumVari,PCs,lamda,confr);
% [T2_0, Q_0, F_0,T2lim,SPElim,Flim,Cw0,D0,fai0] = PCA_kde_sta(x_0,P,NumVari,PCs,lamda,confr);
[NumSampTest,NumVariTest]= size(x_0);

time= 1:1:NumSampTest;
figure;
subplot(3,1,1),semilogy(time,T2_0);
%title('Testing Data: T^2 statistics');
ylabel('T^2')
hold on;
plot(time,T2lim*ones(1,NumSampTest),'r:');
subplot(3,1,2),semilogy(time,Q_0);hold on;
plot(time,SPElim*ones(1,NumSampTest),'r:');
ylabel('SPE')
xlabel('Train healthy Samples')
subplot(3,1,3),semilogy(time,F_0);hold on;
plot(time,Flim*ones(1,NumSampTest),'r:');
ylabel('Combined')
xlabel('Train healthy Samples')

FAR_T2 = sum(T2_0>T2lim)/length(T2_0);
FAR_SPE = sum(Q_0>SPElim)/length(Q_0);
FAR_F = sum(F_0>Flim)/length(F_0);

[CP_T2,CP_SPE,CP_F,RBC_T2,RBC_SPE,RBC_F] = CP_RBC(NumVari,NumSampTest,x_0,D0,Cw0,fai0);
CP_T20 = CP_T2;CP_SPE0 = CP_SPE;CP_F0=CP_F;RBC_T20=RBC_T2;RBC_SPE0=RBC_SPE;RBC_F0=RBC_F;
figure;
subplot(2,2,1)
imagesc(abs(CP_T2'));ylabel('CP  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(abs(CP_SPE'));ylabel('CP  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(abs(CP_F'));ylabel('CP  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_0)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)

figure;
subplot(2,2,1)
imagesc(RBC_T2');ylabel('RBC  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(RBC_SPE');ylabel('RBC  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(RBC_F');ylabel('RBC  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_0)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)

%%
x_1= X_normal_valid;%   Testing data, fault operation
[T2_1, Q_1, F_1,T2lim,SPElim,Flim,Cw1,D1,fai1] = PCA_sta(x_1,P,NumVari,PCs,lamda,confr);
% [T2_1, Q_1, F_1,T2lim,SPElim,Flim,Cw1,D1,fai1] = PCA_kde_sta(x_1,P,NumVari,PCs,lamda,confr);
[NumSampTest,NumVariTest]= size(x_1);

time= 1:1:NumSampTest;
figure;
subplot(3,1,1),semilogy(time,T2_1);
%title('Testing Data: T^2 statistics');
ylabel('T^2')
hold on;
plot(time,T2lim*ones(1,NumSampTest),'r:');
subplot(3,1,2),semilogy(time,Q_1);hold on;
plot(time,SPElim*ones(1,NumSampTest),'r:');
ylabel('SPE')
xlabel('Train Unhealthy Samples')
subplot(3,1,3),semilogy(time,F_1);hold on;
plot(time,Flim*ones(1,NumSampTest),'r:');
ylabel('Combined')
xlabel('Train Unhealthy Samples')

FDR_T2 = sum(T2_1>T2lim)/length(T2_1);
FDR_SPE = sum(Q_1>SPElim)/length(Q_1);
FDR_F = sum(F_1>Flim)/length(F_1);

[CP_T2,CP_SPE,CP_F,RBC_T2,RBC_SPE,RBC_F] = CP_RBC(NumVari,NumSampTest,x_1,D1,Cw1,fai1);
CP_T21 = CP_T2;CP_SPE1 = CP_SPE;CP_F1=CP_F;RBC_T21=RBC_T2;RBC_SPE1=RBC_SPE;RBC_F1=RBC_F;
figure;
subplot(2,2,1)
imagesc(abs(CP_T2'));ylabel('CP  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(abs(CP_SPE'));ylabel('CP  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(abs(CP_F'));ylabel('CP  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_1)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)

figure;
subplot(2,2,1)
imagesc(RBC_T2');ylabel('RBC  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(RBC_SPE');ylabel('RBC  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(RBC_F');ylabel('RBC  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_1)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)

%%
x_2= X_abnormal_valid;%   Testing data, normal operation
[T2_2, Q_2, F_2,T2lim,SPElim,Flim,Cw2,D2,fai2] = PCA_sta(x_2,P,NumVari,PCs,lamda,confr);
% [T2_2, Q_2, F_2,T2lim,SPElim,Flim,Cw2,D2,fai2] = PCA_kde_sta(x_2,P,NumVari,PCs,lamda,confr);
[NumSampTest,NumVariTest]= size(x_2);

time= 1:1:NumSampTest;
figure;
subplot(3,1,1),semilogy(time,T2_2);
%title('Testing Data: T^2 statistics');
ylabel('T^2')
hold on;
plot(time,T2lim*ones(1,NumSampTest),'r:');
subplot(3,1,2),semilogy(time,Q_2);hold on;
plot(time,SPElim*ones(1,NumSampTest),'r:');
ylabel('SPE')
xlabel('Test healthy Samples')
subplot(3,1,3),semilogy(time,F_2);hold on;
plot(time,Flim*ones(1,NumSampTest),'r:');
ylabel('Combined')
xlabel('Test healthy Samples')

tFAR_T2 = sum(T2_2>T2lim)/length(T2_2);
tFAR_SPE = sum(Q_2>SPElim)/length(Q_2);
tFAR_F = sum(F_2>Flim)/length(F_2);

[CP_T2,CP_SPE,CP_F,RBC_T2,RBC_SPE,RBC_F] = CP_RBC(NumVari,NumSampTest,x_2,D2,Cw2,fai2);
CP_T22 = CP_T2;CP_SPE2 = CP_SPE;CP_F2=CP_F;RBC_T22=RBC_T2;RBC_SPE2=RBC_SPE;RBC_F2=RBC_F;
figure;
subplot(2,2,1)
imagesc(abs(CP_T2'));ylabel('CP  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(abs(CP_SPE'));ylabel('CP  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(abs(CP_F'));ylabel('CP  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_2)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)
title('x_2')
figure;
subplot(2,2,1)
imagesc(RBC_T2');ylabel('RBC  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(RBC_SPE');ylabel('RBC  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(RBC_F');ylabel('RBC  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_2)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)
title('x_2')
%%
x_3= X_abnormal_invalid;%   Testing data, fault operation
[T2_3, Q_3, F_3,T2lim,SPElim,Flim,Cw3,D3,fai3] = PCA_sta(x_3,P,NumVari,PCs,lamda,confr);
% [T2_3, Q_3, F_3,T2lim,SPElim,Flim,Cw3,D3,fai3] = PCA_kde_sta(x_3,P,NumVari,PCs,lamda,confr);
% x_1= X_fault_valid;
% x_1= X_fault_test;
[NumSampTest,NumVariTest]= size(x_3);

time= 1:1:NumSampTest;
figure;
subplot(3,1,1),semilogy(time,T2_3);
%title('Testing Data: T^2 statistics');
ylabel('T^2')
hold on;
plot(time,T2lim*ones(1,NumSampTest),'r:');
subplot(3,1,2),semilogy(time,Q_3);hold on;
plot(time,SPElim*ones(1,NumSampTest),'r:');
ylabel('SPE')
xlabel('Test Unhealthy Samples')
subplot(3,1,3),semilogy(time,F_3);hold on;
plot(time,Flim*ones(1,NumSampTest),'r:');
ylabel('Combined')
xlabel('Test Unhealthy Samples')

tFDR_T2 = sum(T2_3>T2lim)/length(T2_3);
tFDR_SPE = sum(Q_3>SPElim)/length(Q_3);
tFDR_F = sum(F_3>Flim)/length(F_3);

b_train_acc =(1-FAR_F+FDR_F)/2;

fprintf('confr is %.4f, sum_pcs is %.4f\nBalanced train acc is %.4f (H:%.4f, N:%.4f)\n',confr,sum_pcs,b_train_acc,1-FAR_F,FDR_F)
b_test_acc =(1-tFAR_F+tFDR_F)/2;
fprintf('Balanced test acc is %.4f (H:%.4f, N:%.4f)\n',b_test_acc,1-tFAR_F,tFDR_F)


%% FDD with contribution plot
[CP_T2,CP_SPE,CP_F,RBC_T2,RBC_SPE,RBC_F] = CP_RBC(NumVari,NumSampTest,x_3,D3,Cw3,fai3);
CP_T23 = CP_T2;CP_SPE3 = CP_SPE;CP_F3=CP_F;RBC_T23=RBC_T2;RBC_SPE3=RBC_SPE;RBC_F3=RBC_F;
figure;
subplot(2,2,1)
imagesc(abs(CP_T2'));ylabel('CP  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(abs(CP_SPE'));ylabel('CP  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(abs(CP_F'));ylabel('CP  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_3)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)

figure;
subplot(2,2,1)
imagesc(RBC_T2');ylabel('RBC  of  T^2','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,2)
imagesc(RBC_SPE');ylabel('RBC  of  SPE','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,3)
imagesc(RBC_F');ylabel('RBC  of  Com','fontsize',18);
xlabel('Samples','fontsize',18)
subplot(2,2,4)
imagesc(abs(x_3)');ylabel('Groudtruth','fontsize',18);
xlabel('Samples','fontsize',18)