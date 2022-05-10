clc;clear; close all;
global  dupx1 dlowx1 dupx2 dlowx2 m2 m1 m3 clow1 cup1
%% Tip-2
a=1;
dup=[-3.5 -a -a 0
    -a 0 0 a
    0 a a 3.5];
dlow=[-3.5 -a -a 0
    -a 0 0 a
    0 a a 3.5];
clow1=[-1 0 1];
cup1=clow1;
d1= dup;
dupx1=dup;dupx2=dup;
dlowx1=dlow;dlowx2=dlow;
%%
k=1;
m1=0.9; m2=1-m1;  m3=m1;
for i=-1:0.0001:1
    
    [yz1(k), y(k)]=tippp2(i);
%     m2=.5; m1=0.1; m3=m1;
%     [yz2(k), y(k)]=tippp2(i);
% %     m2=.9; m1=0.1; m3=m1;
%     [yz3(k), y(k)]=tippp21(i);
    u(k)=i;
    k=k+1;
end
plot(u,yz1,'red','Linewidth',2)
hold on;
plot(u,y,'blue','Linewidth',2)
%%
figure

plot(diff(yz1)/0.0001,'red','Linewidth',2)

% plot(u,yz3,'blue','Linewidth',2)
% %%
% % figure
% plot(u,yz1,'red','Linewidth',2)
% hold on;
% plot(u,(yz1+yz2*0.5)/1.5,'black','Linewidth',2)
% plot(u,(yz1+yz2*0.66+yz2*0.33)/2,'blue','Linewidth',4,'LineStyle','--')

grid on
%%
