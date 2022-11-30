clear all
close all
%%
cd ..

%% add path
addpath('./common_tool');


%% load data and init
% dataset mat file should include fea, gnd

mtds = 2;epss=1e-5;
fsname = 'gmhi';
features = 50;
dataset = 'h_train_gmhi';
load(['./data/',dataset,'.mat']);
dataset2 = 'h_test_gmhi';
load(['./data/',dataset2,'.mat']);


% fsname = 'smg';
% features = 52;
% dataset = 'h_train_smg';
% load(['./data/',dataset,'.mat']);
% dataset2 = 'h_test_smg';
% load(['./data/',dataset2,'.mat']);
% 
% dataset = 'h_train_lgb_v200';
% load(['./data/',dataset,'.mat']);
% dataset2 = 'h_test_lgb_v200';
% load(['./data/',dataset2,'.mat']);
% 
% dataset = 'h_train_cat_v150';
% load(['./data/',dataset,'.mat']);
% dataset2 = 'h_test_cat_v150';
% load(['./data/',dataset2,'.mat']);
% eps=1e-20;
%  X=cell2num2(X_train);
% Xa2=log(X.*(X>0.5)+1);
% Xb=log(log(X.*(X<=0.5)+1)+eps);
% X_train = Xa2.*(X>0.5) + Xb.*(X<=0.5);
% 
% X=X_test;
% Xa2=log(X.*(X>0.5)+1);
% Xb=log(log(X.*(X<=0.5)+1)+eps);
% X_test = Xa2.*(X>0.5) + Xb.*(X<=0.5);
% 
% fea_h = X_train(y_train==0,:);
% fea_n = X_train(y_train==1,:);
% fea = X_train;
% fea2 = log(X_test+1);
% mask_h = (X_train(y_train==0,:))==0;
% mask_n = (X_train(y_train==1,:))==0;
% % fea_h = fea_h;
% % fea_n = log(fea_n+eps);
% % fea = log(fea+eps);
% fea2 = X_test;%log(+eps);
% fea_h(mask_h)=0;
% fea_n(mask_n)=0;
% mask_test = X_test==0;
% fea2(mask_test)=0;
% [fea_h,fea_n,meanxapp,stdxapp] = normalizemeanstd(fea_h,fea_n);
% [~,fea2,~,~] = normalizemeanstd(fea_h,fea2,meanxapp,stdxapp);
%%
X=cell2num2(X_train);
fea_h50_o = X(y_train==0,:);
fea_n50_o = X(y_train==1,:);

% X_train = log2(mtds*X+epss);
% 
% X=X_test;
% X_test = log2(mtds*X+epss);

X_train = X;
for i = 1:size(X,1)
    for j = 1:size(X,2)
        if X(i,j)<=1
            X_train(i,j) = log2(mtds*(X(i,j))+epss);      
        else
            X_train(i,j) = sqrt(X(i,j)); 
        end
    end
end

X=X_test;
% X_test = log2(mtds*(X)+epss);
for i = 1:size(X,1)
    for j = 1:size(X,2)
        if X(i,j)<=1
            X_test(i,j) = log2(mtds*(X(i,j))+epss);            
        else
             X_test(i,j) = sqrt(X(i,j));
        end
    end
end

fea_h = X_train(y_train==0,:);
fea_n = X_train(y_train==1,:);
fea = X_train;

% fea2 = log(X_test+1);
% mask_h = (X_train(y_train==0,:))==0;
% mask_n = (X_train(y_train==1,:))==0;
%% 
 
mask_h = y_train==0;
mask_n = y_train==1;
% fea_h = fea_h;
% fea_n = log(fea_n+eps);
% fea = log(fea+eps);
fea2 = X_test;%log(+eps);
[fea_h,fea_n,meanxapp,stdxapp] = normalizemeanstd(fea_h,fea_n);
fea_h50=fea_h;
fea_n50=fea_n;
[~,fea2,~,~] = normalizemeanstd(fea_h,fea2,meanxapp,stdxapp);
%%
gnd = double(y_train)';
class_num = length(unique(gnd));

if ~iscell(feature_name)        
    feature_name=cellstr(feature_name); 
    fea_names = feature_name;
end

N = size(fea,2);
%% 
load(['./PCA_results/PCA_result_all_',fsname,'_315.mat'])
% features = 10:10:200;
% features = 50;

imax = 0;
for i1 = 1:length(features)
    for i2 = 1:length(0.1:0.1:0.9)
        for i3 = 1:length(0.6:0.1:0.9) %表示以0.6为起点,以0.9为终点,以0.1为步长的一维矩阵
%             tmp = b_train_acc_all_F(i1,i2,i3)+b_test_acc_all_F(i1,i2,i3);
            tmp = (1-b_train_test_all(i1,i2,i3,9))*0.5+b_train_test_all(i1,i2,i3,10)*0.5; %1*9*4*12
%             tmp = (1-b_train_test_all(i1,i2,i3,3))*0.4+b_train_test_all(i1,i2,i3,4)*0.6;
%             tmp = 1-b_train_acc_all_F(i1,i2,i3,1)+b_train_acc_all_F(i1,i2,i3,2);
%             tmp = b_train_acc_all_T2(i1,i2,i3)*0.5+(b_train_acc_all_SPE(i1,i2,i3)+b_train_acc_all_F(i1,i2,i3))*0.5;
            if tmp>imax
                imax = tmp;
                imax1 = i1;imax2=i2;imax3=i3;
            end
        end
    end
end
% fprintf('Best average perform is %.4f,b_train is %.4f,b_test is %.4f\n',imax/2,b_train_acc_all_F(imax1,imax2,imax3),b_test_acc_all_F(imax1,imax2,imax3));
% squeeze(train_acc_F(imax1,imax2,imax3,:)),squeeze(test_acc_F(imax1,imax2,imax3,:))
% fprintf('Best average perform is %.4f,b_train is %.4f,b_test is %.4f\n',imax/2,b_train_acc_all(imax1,imax2,imax3),b_test_acc_all(imax1,imax2,imax3));
% squeeze(train_acc(imax1,imax2,imax3,:)),squeeze(test_acc(imax1,imax2,imax3,:))
num_f=features(imax1);
tmp1 = 0.1:0.1:0.9;
confr=tmp1(imax2);
tmp2=0.6:0.1:0.9;
sum_pcs=tmp2(imax3);
fprintf('Now computing features %d,confr %.4f,sum_PCs %.4f\n',num_f,confr,sum_pcs);
PCA_iter2;
%% 
fea_hipca=fea_h(F_0<Flim,:);
figure
title('hiPCA health prediction')
[NumSampTest,NumVariTest]= size(x_0);
time= 1:1:NumSampTest;
subplot(2,2,1),semilogy(time,Q_0,'.');hold on;
plot(time,SPElim*ones(1,NumSampTest),'Color','red');
ylabel('Q')
xlabel('Healthy Samples(Train set)')
xlim([0 NumSampTest])

[NumSampTest,NumVariTest]= size(x_1);
time= 1:1:NumSampTest;
subplot(2,2,2),semilogy(time,Q_1,'.');hold on;
plot(time,SPElim*ones(1,NumSampTest),'Color','r');
ylabel('Q')
xlabel('Unhealthy Samples(Train set)')
xlim([0 NumSampTest])

[NumSampTest,NumVariTest]= size(x_2);
time= 1:1:NumSampTest;
subplot(2,2,3),semilogy(time,Q_2,'.');hold on;
plot(time,SPElim*ones(1,NumSampTest),'Color','r');
ylabel('Q')
xlabel('Healthy Samples(Test set)')
xlim([0 NumSampTest])

[NumSampTest,NumVariTest]= size(x_3);
time= 1:1:NumSampTest;
subplot(2,2,4),semilogy(time,Q_3,'.');hold on;
plot(time,SPElim*ones(1,NumSampTest),'Color','r');
ylabel('Q')
xlabel('Unhealthy Samples(Test set)')
xlim([0 NumSampTest])
%
%% Correlation matrix
meta_iPCA=zeros(size([F_0;F_1]'));meta_iPCA(mask_h)=F_0;meta_iPCA(mask_n)=F_1;
load('./data/h_train_all_meta.mat')

meta_all = [meta_age;meta_bmi;meta_TRIG;meta_CHOL;meta_fbg;meta_HDLC;meta_LDLC;num2cell(meta_iPCA)];

figure
dat_table = cell2num(meta_all)';%
ind_set = [1:8];
sets = {'AGE','BMI','TRIG','CHOL','FBG','HDLC','LDLC','hiPCA'};%dat_table(:,6) = dat_table(:,6)*38.67;
[R,Pval,H] = corrplot(dat_table(:,ind_set),'varNames',sets(ind_set),'type','Spearman','rows','pairwise','testR','on','alpha',0.001)
1;
%% Regional pies
load('./data/X_genus_712.mat')
wordcloud(meta_country);
country_names=meta_country;
country_names2 = meta_country(mask_h);
country_names_h=country_names2(F_0<Flim);
country_names_h=strrep(country_names_h,'China','Chinese');
country_names_h=strrep(country_names_h,'Danish population','Danish');
country_names_h=strrep(country_names_h,'danish','Danish');
country_names_h=strrep(country_names_h,'Mongolian Population','Mongolian');
country_names_h=strrep(country_names_h,'Sardinian Population','Sardinian');
% load X_genus_F0_hipca2.mat
country_names_h1 = country_names_h(clusterX==1);
country_names_h2 = country_names_h(clusterX==2);
country_names_h3 = country_names_h(clusterX==3);
country_names_h4 = country_names_h(clusterX==4);
figure
subplot(2,2,1)
pie(categorical(country_names_h1))
subplot(2,2,2)
pie(categorical(country_names_h2))
subplot(2,2,3)
pie(categorical(country_names_h3))
subplot(2,2,4)
pie(categorical(country_names_h4))
clear count
uni_cname = unique(country_names_h);
for i = 1:length(uni_cname)
    uni_cnum(1,i) = sum(count(country_names_h1,uni_cname{1,i}));
    uni_cnum(2,i) = sum(count(country_names_h2,uni_cname{1,i}));
    uni_cnum(3,i) = sum(count(country_names_h3,uni_cname{1,i}));
    uni_cnum(4,i) = sum(count(country_names_h4,uni_cname{1,i}));
end
uni_cnums = sum(uni_cnum);
ind100 = find(uni_cnums>=50);
indlen = ceil(length(ind100)/2);
figure
for i = 1:length(ind100)
    subplot(2,indlen,i)
    pie(uni_cnum(:,ind100(i)))
    title(uni_cname(ind100(i)))
    if i == 1
        legend('HP1','HP2','HP3','HP4')
    end
end
%% Boxplota and Bar: iPCA values & Acc values in Train data
idx_all = 1:size(fea,1);
idx_train_healthy = find(strcmp(meta_train, 'Healthy' ));
idx_train_unhealthy = setdiff(idx_all,idx_train_healthy);
idx_ACVD = find(strcmp(meta_train, 'ACVD' ));                              % id_1 = strcmp(meta_train, 'ACVD' );
idx_CRC = find(strcmp(meta_train, 'CRC' ));                                % id_2 = strcmp(meta_train, 'CRC' );
idx_CD = find(strcmp(meta_train, 'Crohns disease' ));                      % id_3 = strcmp(meta_train, 'Crohns disease');
idx_IGT = find(strcmp(meta_train, 'IGT' ));                                % id_4 = strcmp(meta_train,'IGT');
idx_Obesity = find(strcmp(meta_train, 'Obesity' ));                        % id_5 = strcmp(meta_train,'Obesity');
idx_Overweight = find(strcmp(meta_train, 'Overweight' ));                  % id_6 = strcmp(meta_train,'Overweight');
idx_RA = find(strcmp(meta_train, 'Rheumatoid Arthritis' ));                % id_7 = strcmp(meta_train,'Rheumatoid Arthritis');
idx_SA = find(strcmp(meta_train, 'Symptomatic atherosclerosis' ));         % id_8 = strcmp(meta_train,'Symptomatic atherosclerosis');
idx_T2D = find(strcmp(meta_train, 'T2D' ));                                % id_9 = strcmp(meta_train,'T2D');   
idx_UC = find(strcmp(meta_train, 'Ulcerative colitis' ));                  % id_10 = strcmp(meta_train,'Ulcerative colitis');  
idx_Underweight = find(strcmp(meta_train, 'Underweight' ));                % id_11 = strcmp(meta_train,'Underweight');                      
idx_aa = find(strcmp(meta_train, 'advanced adenoma' ));                    % id_12 = strcmp(meta_train,'advanced adenoma');  

% jj = 1;
% for i = 1:length(meta_train)
%     if  strcmp(meta_train(i), 'ACVD' )
%         idss(jj) = 1;
%     elseif strcmp(meta_train(i), 'CRC' )
%         idss(jj) = 2;
%     elseif strcmp(meta_train(i), 'Crohns disease' )
%         idss(jj) = 3;
%     elseif strcmp(meta_train(i), 'IGT ')
%         idss(jj) = 4;
%     elseif strcmp(meta_train(i), 'Obesity' )
%         idss(jj) = 5;
%     elseif strcmp(meta_train(i), 'Overweight' )
%         idss(jj) = 6;
%     elseif strcmp(meta_train(i), 'Rheumatoid Arthritis')
%         idss(jj) = 7;
%     elseif strcmp(meta_train(i), 'Symptomatic atherosclerosis' )
%         idss(jj) = 8;
%     elseif strcmp(meta_train(i), 'T2D' )
%         idss(jj) = 9;
%     elseif strcmp(meta_train(i), 'Ulcerative colitis' )
%         idss(jj) = 10;
%     elseif strcmp(meta_train(i), 'Underweight')
%         idss(jj) = 11;
%     elseif strcmp(meta_train(i), 'advanced adenoma' )
%         idss(jj) = 12;
%     elseif strcmp(meta_train(i), 'Healthy' )
%         continue;
%     end
%     jj=jj+1;
% end

iPCA_Heathy = F_0;
FAR_Healthy = length(find(iPCA_Heathy>Flim))/length(F_0);
FDR_Healthy = length(find(iPCA_Heathy<Flim))/length(F_0);

c_ACVD=ismember(idx_train_unhealthy,idx_ACVD);
iPCA_ACVD = F_1(c_ACVD);
FDR_ACVD = length(find(iPCA_ACVD>Flim))/length(F_1(c_ACVD));

c_CRC=ismember(idx_train_unhealthy,idx_CRC);
iPCA_CRC = F_1(c_CRC);
FDR_CRC = length(find(iPCA_CRC>Flim))/length(F_1(c_CRC));
 
c_CD=ismember(idx_train_unhealthy,idx_CD);
iPCA_CD = F_1(c_CD);
FDR_CD = length(find(iPCA_CD>Flim))/length(F_1(c_CD));

c_IGT=ismember(idx_train_unhealthy,idx_IGT);
iPCA_IGT = F_1(c_IGT);
FDR_IGT = length(find(iPCA_IGT>Flim))/length(F_1(c_IGT));

c_Obesity=ismember(idx_train_unhealthy,idx_Obesity);
iPCA_Obesity = F_1(c_Obesity);
FDR_Obesity = length(find(iPCA_Obesity>Flim))/length(F_1(c_Obesity));

c_Overweight=ismember(idx_train_unhealthy,idx_Overweight);
iPCA_Overweight = F_1(c_Overweight);
FDR_Overweight = length(find(iPCA_Overweight>Flim))/length(F_1(c_Overweight));
 
c_RA=ismember(idx_train_unhealthy,idx_RA);
iPCA_RA = F_1(c_RA);
FDR_RA = length(find(iPCA_RA>Flim))/length(F_1(c_RA));

c_SA=ismember(idx_train_unhealthy,idx_SA);
iPCA_SA = F_1(c_SA);
FDR_SA = length(find(iPCA_SA>Flim))/length(F_1(c_SA));

c_T2D=ismember(idx_train_unhealthy,idx_T2D);
iPCA_T2D = F_1(c_T2D);
FDR_T2D = length(find(iPCA_T2D>Flim))/length(F_1(c_T2D));
  
c_UC=ismember(idx_train_unhealthy,idx_UC);
iPCA_UC = F_1(c_UC);
FDR_UC = length(find(iPCA_UC>Flim))/length(F_1(c_UC));

c_Underweight=ismember(idx_train_unhealthy,idx_Underweight);
iPCA_Underweight = F_1(c_Underweight);
FDR_Underweight = length(find(iPCA_Underweight>Flim))/length(F_1(c_Underweight));

c_aa=ismember(idx_train_unhealthy,idx_aa);
iPCA_aa = F_1(c_aa);
FDR_aa = length(find(iPCA_aa>Flim))/length(F_1(c_aa));

figure

g1 = repmat({'Healthy'},length(idx_train_healthy),1);
g2 = repmat({'ACVD'},length(idx_ACVD),1);
g3 = repmat({'CRC'},length(idx_CRC),1);
g4 = repmat({'CD'},length(idx_CD),1);
g5 = repmat({'IGT'},length(idx_IGT),1);
g6 = repmat({'Obesity'},length(idx_Obesity),1);
g7 = repmat({'Overweight'},length(idx_Overweight),1);
g8 = repmat({'RA'},length(idx_RA),1);
g9 = repmat({'SA'},length(idx_SA),1);
g10 = repmat({'T2D'},length(idx_T2D),1);
g11 = repmat({'UC'},length(idx_UC),1);
g12 = repmat({'Underweight'},length(idx_Underweight),1);
g13 = repmat({'advanced adenoma'},length(idx_aa),1);
g = [g1; g2; g3; g4; g5; g6; g7; g8; g9; g10; g11; g12; g13];
HH = boxplot([iPCA_Heathy;iPCA_ACVD;iPCA_CRC;iPCA_CD;iPCA_IGT;iPCA_Obesity;iPCA_Overweight;iPCA_RA;iPCA_SA;iPCA_T2D;iPCA_UC;iPCA_Underweight;iPCA_aa],g);
xtickangle(45)
ylabel('hiPCA')
title('Compare cohorts')

pvs = zeros(1,12);
[h,pvs(1),ci,stats] = ttest2(iPCA_Heathy,iPCA_ACVD);
[h,pvs(2),ci,stats] = ttest2(iPCA_Heathy,iPCA_CRC);
[h,pvs(3),ci,stats] = ttest2(iPCA_Heathy,iPCA_CD);
[h,pvs(4),ci,stats] = ttest2(iPCA_Heathy,iPCA_IGT);
[h,pvs(5),ci,stats] = ttest2(iPCA_Heathy,iPCA_Obesity);
[h,pvs(6),ci,stats] = ttest2(iPCA_Heathy,iPCA_Overweight);
[h,pvs(7),ci,stats] = ttest2(iPCA_Heathy,iPCA_RA);
[h,pvs(8),ci,stats] = ttest2(iPCA_Heathy,iPCA_SA);
[h,pvs(9),ci,stats] = ttest2(iPCA_Heathy,iPCA_T2D);
[h,pvs(10),ci,stats] = ttest2(iPCA_Heathy,iPCA_UC);
[h,pvs(11),ci,stats] = ttest2(iPCA_Heathy,iPCA_Underweight);
[h,pvs(12),ci,stats] = ttest2(iPCA_Heathy,iPCA_aa);
groups={{'Healthy','ACVD'},...
 		{'Healthy','CRC'},... %note you can mix and match notions
		{'Healthy','CD'},...
        {'Healthy','IGT'},...
        {'Healthy','Obesity'},...
        {'Healthy','Overweight'},...
        {'Healthy','RA'},...
        {'Healthy','SA'},...
        {'Healthy','T2D'},...
        {'Healthy','UC'},...
        {'Healthy','Underweight'},...
        {'Healthy','advanced adenoma'}
        };

HH=sigstar(groups,pvs);    

figure

c = categorical({'Healthy','ACVD','CRC','CD','IGT','Obesity','Overweight','RA','SA','T2D','UC','Underweight','advanced adenoma'});
FDRs = [FDR_Healthy FDR_ACVD FDR_CRC FDR_CD FDR_IGT FDR_Obesity FDR_Overweight FDR_RA FDR_SA FDR_T2D FDR_UC FDR_Underweight FDR_aa];
b=bar(c,FDRs);
b.FaceColor = 'flat';
b.LineWidth = 1.5
b.CData(4,:) = [.5 0 .5];
ylabel('hiPCA accuracy ratio')


%%
if 0
    figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F0),'k')
    ylabel('Healthy(RBC with Combined)')
%     figure
    % subplot(1,3,1)
    % bar(categorical(feature_name(1:num_f)),mean(RBC_T21(c_ACVD,:)),'k')
    % ylabel('ACVD(RBC with T^2)')
    % subplot(1,3,2)
    % bar(categorical(feature_name(1:num_f)),mean(RBC_SPE1(c_ACVD,:)),'k')
    % ylabel('ACVD(RBC with SPE)')
    % subplot(1,3,3)
    figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_ACVD,:)),'k')
    ylabel('ACVD(RBC with Combined)')
    figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_CRC,:)),'k')
    ylabel('CRC(RBC with Combined)')
    figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_CD,:)),'k')
    ylabel('CD(RBC with Combined)')
     figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_IGT,:)),'k')
    ylabel('IGT(RBC with Combined)')
     figure
    bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_Obesity,:)),'k')
    ylabel('Obesity(RBC with Combined)')
end
% close all
feature_names = fea_names; 
figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_name(1:num_f)),mean(RBC_F1(c_ACVD,:)),w2,'FaceColor',[1 0 0])
% bar(categorical(feature_names(1:num_f)),median(RBC_F1(c_ACVD,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('ACVD')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_CRC,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('CRC')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_CD,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('CD')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_IGT,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('IGT')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_Obesity,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('Obesity')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_Overweight,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('Overweight')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_RA,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('RA')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_SA,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('SA')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_T2D,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('T2D')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_UC,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('UC')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_Underweight,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('UW')

figure
w1 = 0.7; 
bar(categorical(feature_names(1:num_f)),mean(RBC_F0),w1,'FaceColor',[0.2 0.2 0.5])
w2 = .3;
hold on
bar(categorical(feature_names(1:num_f)),mean(RBC_F1(c_aa,:)),w2,'FaceColor',[1 0 0])
hold off
ylabel('RBC from \phi')
title('advanced adenoma')


c_sets = {c_ACVD;c_CRC;c_CD;c_IGT;c_Obesity;c_Overweight;c_RA;c_SA;c_T2D;c_UC;c_Underweight;c_aa};
Top_N = num_f;
for i = 1:length(c_sets)
   Fmeans = median(RBC_F1(c_sets{i},:));
   [indvals,indx] = sort(Fmeans,'descend');
   join_name = join(feature_names(indx(1:Top_N)),';');
   fprintf('Top %d biomkers for %s are %s\n',Top_N, string(c(i+1)),join_name{1,1});
end
