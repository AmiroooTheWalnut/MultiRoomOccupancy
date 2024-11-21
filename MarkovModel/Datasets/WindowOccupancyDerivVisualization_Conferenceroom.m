clc
clear
data=readtable(strcat('Preprocessed_Conferenceroomfrom19June2023to30June2023.csv'));
data=table2array(data);
deriv=diff(data(:,11));
data=data(2:size(data,1),:);
for i=1:size(data,1)
    if data(i,15)>2
        data(i,15)=2;
    end
end
XOcc=data(:,15);
ColorWin=data(:,14);
figure(1)
clf
scatter(XOcc,deriv,[],ColorWin)
