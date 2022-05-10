clc;clear all;close all;
global  dupx1 dlowx1 dupx2 dlowx2 m2 m1 m3 clow1 cup1
%% Tip-2
a=1;
m2=0.5;
m1=1-m2;
m3=m1;
dup=[-3.5 -a -a 0
    -a 0 0 a
    0 a a 3.5];
dlow=[-3.5 -a -a 0
    -a 0 0 a
    0 a a 3.5];
clow1=[-.1 0 1];
cup1=[-.1 0 1];
d1= dup;
dupx1=dup;dupx2=dup;
dlowx1=dlow;dlowx2=dlow;
%%
k=1;
for i=-1:0.01:1
    [y(k), y1(k)]=tippp2(i);
%     y2(k)=evalfis(i,T1);
    u(k)=i;
    k=k+1;
end

% figure
plot(u,y1,'black','Linewidth',2)
hold on;
% plot(u,y1,'black','Linewidth',2,'LineStyle','--')

plot(u,y,'red','Linewidth',2)

grid on
% legend('LCC','NCC-3','Location','NorthWest')