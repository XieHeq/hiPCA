function [CP_T2,CP_SPE,CP_F,RBC_T2,RBC_SPE,RBC_F] = CP_RBC(NumVari,NumSampTest,x_1,D,Cw,fai)

ksi=eye(NumVari);
CP_T2=zeros(NumSampTest,NumVari);
CP_SPE=zeros(NumSampTest,NumVari);
CP_F=zeros(NumSampTest,NumVari);
for i=1:NumSampTest
    for j=1:NumVari
        CP_T2(i,j)= (x_1(i,:)*(D(j,:)'+D(:,j)))^2;
        %(ksi(:,j)'*D^0.5*x_1(i,:)')^2;
        %x_1(i,:)*D*ksi(:,j)*inv(ksi(:,j)'*D*ksi(:,j))*ksi(:,j)'*D*x_1(i,:)';
        CP_SPE(i,j)= (x_1(i,:)*(Cw(j,:)'+Cw(:,j)))^2;
        %(ksi(:,j)'*Cw^0.5*x_1(i,:)')^2;
        %x_1(i,:)*Cw*ksi(:,j)*inv(ksi(:,j)'*Cw*ksi(:,j))*ksi(:,j)'*Cw*x_1(i,:)';
        CP_F(i,j) = (x_1(i,:)*(fai(j,:)'+fai(:,j)))^2;
%         (ksi(:,j)'*fai^0.5*x_1(i,:)')^2;
        %x_1(i,:)*fai*ksi(:,j)*inv(ksi(:,j)'*fai*ksi(:,j))*ksi(:,j)'*fai*x_1(i,:)';
    end
end

%% RBC by T2 and SPE

ksi=eye(NumVari);
RBC_T2=zeros(NumSampTest,NumVari);
RBC_SPE=zeros(NumSampTest,NumVari);
RBC_F=zeros(NumSampTest,NumVari);
for i=1:NumSampTest
    for j=1:NumVari
        RBC_T2(i,j)=x_1(i,:)*D*ksi(:,j)*pinv(ksi(:,j)'*D*ksi(:,j))*ksi(:,j)'*D*x_1(i,:)';
        RBC_SPE(i,j)=x_1(i,:)*Cw*ksi(:,j)*pinv(ksi(:,j)'*Cw*ksi(:,j))*ksi(:,j)'*Cw*x_1(i,:)';
        RBC_F(i,j) =x_1(i,:)*fai*ksi(:,j)*pinv(ksi(:,j)'*fai*ksi(:,j))*ksi(:,j)'*fai*x_1(i,:)';
    end
end

