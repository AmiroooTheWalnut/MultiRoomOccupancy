clc
clear
data=readtable(strcat('Preprocessed_Meetingroomfrom1May2023to17May2023.csv'));
data=table2array(data);
deriv=diff(data(:,11));
data=data(2:size(data,1),:);
for i=1:size(data,1)
    if data(i,16)>2
        data(i,16)=2;
    end
end
XOcc=data(:,16);
ColorWin=data(:,15);
ColorWin(1,3)=0;
for i=1:size(ColorWin,1)
    if ColorWin(i,1)==0
        ColorWin(i,:)=[1,0,0];
    else
        ColorWin(i,:)=[0,0,1];
    end

end
figure(1)
clf
scatter(XOcc,deriv,[],ColorWin)

DWCUO=[];
DWOUO=[];
for i=1:size(data,1)
    if data(i,15)==0
        if data(i,16)==0
            DWCUO(1,size(DWCUO)+1)=deriv(i,1);
        end
    else
        if data(i,16)==0
            DWOUO(1,size(DWOUO)+1)=deriv(i,1);
        end
    end
end
figure(2)
clf
hold on
histogram(DWCUO)
histogram(DWOUO)
legend({'win closed','win open'})
xlabel("Derivate of CO2")
title("Unoccupied")

DWCO1=[];
DWOO1=[];
for i=1:size(data,1)
    if data(i,15)==0
        if data(i,16)==1
            DWCO1(1,size(DWCO1)+1)=deriv(i,1);
        end
    else
        if data(i,16)==1
            DWOO1(1,size(DWOO1)+1)=deriv(i,1);
        end
    end
end
figure(3)
clf
hold on
histogram(DWCO1)
histogram(DWOO1)
legend({'win closed','win open'})
xlabel("Derivate of CO2")
title("1 Occupancy")

DWCOM=[];
DWOOM=[];
for i=1:size(data,1)
    if data(i,15)==0
        if data(i,16)==2
            DWCOM(1,size(DWCOM)+1)=deriv(i,1);
        end
    else
        if data(i,16)==2
            DWOOM(1,size(DWOOM)+1)=deriv(i,1);
        end
    end
end
figure(4)
clf
hold on
histogram(DWCOM)
histogram(DWOOM)
legend({'win closed','win open'})
xlabel("Derivate of CO2")
title("Multi Occupancy")