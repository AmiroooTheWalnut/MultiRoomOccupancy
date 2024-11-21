clc
clear
data=readtable('artificialData_threeState_Test1.csv');
data=table2array(data);
deriv=data(:,2);
[sortedDeriv,I]=sort(deriv);
colorIndex(1,size(data,1))=0;
allXs=[];
allY1=[];
allY2=[];
allY3=[];
counter1=1;
counter2=1;
counter3=1;
for i=1:size(sortedDeriv,1)
    colorIndex(1,i)=data(I(i,1),3);
    allXs(i)=data(I(i,1),3);
    if data(I(i,1),3)==0
        allY1(counter1)=sortedDeriv(i,1);
        counter1=counter1+1;
    elseif data(I(i,1),3)==1
        allY2(counter2)=sortedDeriv(i,1);
        counter2=counter2+1;
    else
        allY3(counter3)=sortedDeriv(i,1);
        counter3=counter3+1;
    end
end
figure(1)
scatter(allXs,sortedDeriv,[],colorIndex)
figure(2)
clf
plot(data(:,2))
hold on
plot(data(:,3))
figure(3)
clf
hold on
histogram(allY1)
histogram(allY2)
histogram(allY3)
legend('unoccupied','1occupied','multioccupied')

deriv=data(:,4);
[sortedDeriv,I]=sort(deriv);
colorIndex(1,size(data,1))=0;
allXs=[];
allY1=[];
allY2=[];
allY3=[];
counter1=1;
counter2=1;
counter3=1;
for i=1:size(sortedDeriv,1)
    colorIndex(1,i)=data(I(i,1),3);
    allXs(i)=data(I(i,1),3);
    if data(I(i,1),3)==0
        allY1(counter1)=sortedDeriv(i,1);
        counter1=counter1+1;
    elseif data(I(i,1),3)==1
        allY2(counter2)=sortedDeriv(i,1);
        counter2=counter2+1;
    else
        allY3(counter3)=sortedDeriv(i,1);
        counter3=counter3+1;
    end
end
figure(4)
scatter(allXs,sortedDeriv,[],colorIndex)
figure(5)
clf
plot(data(:,4))
hold on
plot(data(:,3))
figure(6)
clf
hold on
histogram(allY1)
histogram(allY2)
histogram(allY3)
legend('unoccupied','1occupied','multioccupied')