clc
clear

data=readtable('Preprocessed_Meetingroomfrom1May2023to17May2023.csv');
header=data.Properties.VariableNames;
data=table2array(data);
currentDay=data(1,3);
summedByDayData(1,16)=0;
sumDay(1,16)=0;
counter=1;
counterForDay=0;
for i=1:size(data,1)
    if currentDay==data(i,3)
        sumDay=sumDay+data(i,:);
        counterForDay=counterForDay+1;
    else
        sumDay=sumDay/counterForDay;
        summedByDayData(counter,:)=sumDay;
        counter=counter+1;
        counterForDay=0;
        sumDay=[];
        sumDay(1,16)=0;
        currentDay=data(i,3);
    end
end
output=array2table(summedByDayData);
output.Properties.VariableNames=header;
writetable(output,'PreprocessedDaily_Meetingroomfrom1May2023to17May2023.csv')