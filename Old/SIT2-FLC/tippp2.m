function [y y1] = tippp2(x1)
global  dupx1 dlowx1 m2 m1 m3 clow1 d1 cup1
% =clow1;
aitu2=trapm(x1, dupx1,3);
aitl=trapm(x1, dlowx1,3).*[m1 m2 m3];
aitu=trapm(x1, dupx1,3);
if aitu2(1)>0
    aitl1=aitl(1:2);
    aitu1=aitu(1:2);
    clow=clow1(1:2);
    cup=cup1(1:2);
else
    aitl1=aitl(2:3);
    aitu1=aitu(2:3);
    clow=clow1(2:3);
    cup=cup1(2:3);
end
yl=(clow(1)*aitu1(1)+clow(2)*aitl1(2))/(aitu1(1)+aitl1(2));
yr=(cup(1)*aitl1(1)+cup(2)*aitu1(2))/(aitl1(1)+aitu1(2));
y=(yl+yr)*0.5;


if aitu2(1)>0
    aitl1=aitu2(1:2);
    aitu1=aitu2(1:2);
    clow=clow1(1:2);
    cup=cup1(1:2);
else
    aitl1=aitu2(2:3);
    aitu1=aitu2(2:3);
    clow=clow1(2:3);
    cup=cup1(2:3);
end
yl=(clow(1)*aitu1(1)+clow(2)*aitl1(2))/(aitu1(1)+aitl1(2));
yr=(cup(1)*aitl1(1)+cup(2)*aitu1(2))/(aitl1(1)+aitu1(2));
y1=(yl+yr)*0.5;