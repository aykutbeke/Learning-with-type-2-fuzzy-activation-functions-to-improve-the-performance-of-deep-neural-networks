function [y1] = tip1(x1)
global d1 c m
aitl=trapm(x1, d1,m);
for i=1:m
    y(i)=aitl(i)*c(i);
end
y1=sum(y);
% aitu=trapm(x1, d1);
% if aitu(1)>0
%     aitl1=aitl(1:2);
%     aitu1=aitu(1:2);
%     c=c(1:2);
% else
%     aitl1=aitl(2:3);
%     aitu1=aitu(2:3);
%     c=c(2:3);
% end
% yl=(c(1)*aitu1(1)+c(2)*aitl1(2))/(aitu1(1)+aitl1(2));
% yr=(c(1)*aitl1(1)+c(2)*aitu1(2))/(aitl1(1)+aitu1(2));
% y1=(yl+yr)*0.5;

end