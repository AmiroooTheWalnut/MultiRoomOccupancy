function [x]=genSol(constraints,temperature,x0)
x(1,size(constraints,2))=0;
for i=1:size(constraints,2)
    range=constraints(2,i)-constraints(1,i);
    x(1,i)=x0(1,i)+(rand(1,1)-0.5)*range*temperature;
    if x(1,i)<constraints(1,i)
        x(1,i)=constraints(1,i);
    end
    if x(1,i)>constraints(2,i)
        x(1,i)=constraints(2,i);
    end
end
if x(1,1)<x(1,2)
    temp=x(1,1);
    x(1,1)=x(1,2);
    x(1,2)=temp;
end
if x(1,3)>x(1,4)
    temp=x(1,3);
    x(1,3)=x(1,4);
    x(1,4)=temp;
end