clc
clear

% PREPROCESS DATA FROM CITYAIR WEBSITE
data=readtable('Conferenceroomfrom17June2023to30June2023.xlsx');
dataProcessed(size(data,1),13)=0;
for i=1:size(data,1)
    y=year(data{i,1});
    dataProcessed(i,1)=y;
    m=month(data{i,1});
    dataProcessed(i,2)=m;
    d=day(data{i,1});
    dataProcessed(i,3)=d;
    h=hour(data{i,1});
    dataProcessed(i,4)=h;
    mi=minute(data{i,1});
    if second(data{i,1})>30
        mi=mi+1;
        if mi==60
            mi=0;
        end
        dataProcessed(i,5)=mi;
    else
        if mi==60
            mi=0;
        end
        dataProcessed(i,5)=mi;
    end
    dataProcessed(i,6)=data{i,2};
    dataProcessed(i,7)=data{i,3};
    dataProcessed(i,8)=data{i,4};
    dataProcessed(i,9)=data{i,5};
    dataProcessed(i,10)=data{i,6};
    dataProcessed(i,11)=data{i,7};

    if strcmp(data{i,8},'Temperature')
        dataProcessed(i,12)=1;
    elseif strcmp(data{i,8},'Pressure')
        dataProcessed(i,12)=2;
    elseif strcmp(data{i,8},'Humidity')
        dataProcessed(i,12)=3;
    elseif strcmp(data{i,8},'PM2.5')
        dataProcessed(i,12)=4;
    elseif strcmp(data{i,8},'PM10')
        dataProcessed(i,12)=5;
    elseif strcmp(data{i,8},'CO2')
        dataProcessed(i,12)=6;
    end
    dataProcessed(i,13)=data{i,9};
end

% PREPROCESS THE LABELS FROM GITHUB
root = cd('..\..\data\2023\06');
files = dir(pwd);
dirFlags = [files.isdir];
dirs = files(dirFlags);
labelDataProcessed(1,7)=0;
counter=1;
for dayVal=3:size(dirs,1)
    month5Root=cd(dirs(dayVal).name);
    dailyLabels=readtable('conference_room_no1.csv');
    for i=1:size(dailyLabels,1)
        y=2023;
        labelDataProcessed(counter,1)=y;
        m=6;
        labelDataProcessed(counter,2)=m;
        d=str2num(dirs(dayVal).name);
        labelDataProcessed(counter,3)=d;
        h=floor(hours(dailyLabels{i,1}));
        labelDataProcessed(counter,4)=h;
        mi=mod(floor(minutes(dailyLabels{i,1})),60);
        if mi==60
            mi=0;
        end
        labelDataProcessed(counter,5)=mi;
        
        if strcmp(dailyLabels{i,2},'open')==1
            labelDataProcessed(counter,6)=1;
        else
            labelDataProcessed(counter,6)=0;
        end
        labelDataProcessed(counter,7)=dailyLabels{i,3};

        counter=counter+1;
    end
    cd(month5Root)
end
cd(root)

% CONNECT TWO DATASETS
header={'Year','Month','Day','Hour','Minute','Temperature','Pressure',...
    'Humidity','PM2.5','PM10','CO2','Critical pollutant InstantAQI',...
    'InstantAQI','Door','Occupancy'};
counter=1;
finalDataProcessed(1,15)=0;
lastSensorIndex=1;
for i=1:size(labelDataProcessed)
    for j=lastSensorIndex:size(dataProcessed,1)
        if labelDataProcessed(i,1)==dataProcessed(j,1) && ...
           labelDataProcessed(i,2)==dataProcessed(j,2) && ...
           labelDataProcessed(i,3)==dataProcessed(j,3) && ...
           labelDataProcessed(i,4)==dataProcessed(j,4) && ...
           labelDataProcessed(i,5)==dataProcessed(j,5)
            finalDataProcessed(counter,1:13)=dataProcessed(j,:);
            finalDataProcessed(counter,14)=labelDataProcessed(i,6);
            finalDataProcessed(counter,15)=labelDataProcessed(i,7);
            counter=counter+1;
            lastSensorIndex=j;
            break
        else
            if (labelDataProcessed(i,1)==dataProcessed(j,1) || labelDataProcessed(i,1)<dataProcessed(j,1)) && ...
               (labelDataProcessed(i,2)==dataProcessed(j,2) || labelDataProcessed(i,2)<dataProcessed(j,2)) && ...
               (labelDataProcessed(i,3)==dataProcessed(j,3) || labelDataProcessed(i,3)<dataProcessed(j,3)) && ...
               (labelDataProcessed(i,4)==dataProcessed(j,4) || labelDataProcessed(i,4)<dataProcessed(j,4)) && ...
               (labelDataProcessed(i,5)==dataProcessed(j,5) || labelDataProcessed(i,5)<dataProcessed(j,5))
                break
            elseif (labelDataProcessed(i,1)==dataProcessed(j,1) || labelDataProcessed(i,1)>dataProcessed(j,1)) && ...
               (labelDataProcessed(i,2)==dataProcessed(j,2) || labelDataProcessed(i,2)>dataProcessed(j,2)) && ...
               (labelDataProcessed(i,3)==dataProcessed(j,3) || labelDataProcessed(i,3)>dataProcessed(j,3)) && ...
               (labelDataProcessed(i,4)==dataProcessed(j,4) || labelDataProcessed(i,4)>dataProcessed(j,4)) && ...
               (labelDataProcessed(i,5)==dataProcessed(j,5) || labelDataProcessed(i,5)>dataProcessed(j,5))
                lastSensorIndex=j;
            end
        end
    end
end

tempPlotData=finalDataProcessed;
tempPlotData(:,1)=tempPlotData(:,1)/2000;
tempPlotData(:,5)=tempPlotData(:,5)/60;
tempPlotData(:,7)=tempPlotData(:,7)/700;
tempPlotData(:,9)=tempPlotData(:,9)/2;
tempPlotData(:,10)=tempPlotData(:,10)/1600;
tempPlotData(:,11)=tempPlotData(:,11)/400;
tempPlotData(:,15)=tempPlotData(:,15)*32;
plot(tempPlotData,'DisplayName','tempPlotData')


tableData = array2table(finalDataProcessed);
tableData.Properties.VariableNames = header;
writetable(tableData,'Preprocessed_Conferenceroomfrom19June2023to30June2023.csv')