clear; clc; close all

k = 1;
alpha1 = 0.0320;
alpha2 = alpha1;
B1=0.0818;
B2= 0.5736;

for i=-1:0.01:1
    xt(k) = i/1;
    if(xt(k)<0)
         K = t2Gain(xt(k), alpha1, B1);
         xt2(k) = xt(k)*K*1;
    else
         K = t2Gain(xt(k), alpha2, B2);
         xt2(k) = xt(k)*K*1;
    end
    k=k+1;
end

plot(xt*1,xt*1,'black','Linewidth',2)
hold on;
plot(xt*1,xt2,'red','Linewidth',2)
grid on
% figure
% plot(xt,xt,'black','Linewidth',2)
% hold on;
% plot(xt,predict(xt, alpha2, B2, B1),'red','Linewidth',2)
% grid on
% %% Derivatives
syms x sigma alpha1 B1 alpha2  B2 sf

sigma = x*sf;  
K1 = 0.5*(B1/(sigma - sigma*alpha1 + alpha1)...
        - (B1-B1*alpha1)/(sigma*alpha1-1));
    
K2 = 0.5*(B2/(-sigma + sigma*alpha2 + alpha2)...
        - (B2-B2*alpha2)/(-sigma*alpha2-1));

  
f1(x) = sigma*K1*1/sf;
f2(x) = sigma*K2*1/sf;
dfdx1=diff(f1,x); dfdx2=diff(f2,x);
dfda1=diff(f1,alpha1); dfda2=diff(f2,alpha2);
dfdb1=diff(f1,B1); dfdb2=diff(f2,B2);
