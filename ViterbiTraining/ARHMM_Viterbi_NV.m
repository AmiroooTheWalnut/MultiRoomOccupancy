clc
clear
isContinueLearning=1;% If set to 0, the parameters are initialized randomly, if set to 1, the parameters start from the last run that they were saved as a file

% UNCOMMENT EACH PART TO RUN FOR SPECIFIC DATASET

%\/\/\/ SYNTHETIC DATASET
rawData=table2array(readtable('two_room_artificial_med.csv'));
lastSaveSuffix="TR_artificialData_NV_med";
timeFrame=size(rawData,1);
realDoor=rawData(1:timeFrame,3);
obs_CO21=rawData(1:timeFrame,4);
obs_CO21=obs_CO21';
[obs_CO2_2,cost] = tvd_mm(obs_CO21,100,100);
obs_CO2_2=obs_CO2_2';
obs_CO2_2 = smoothdata(obs_CO2_2,"sgolay",100);
% figure(1)
% clf
% hold on
% plot(obs_CO2)
% plot(obs_CO2_2)
obs_CO2_1=obs_CO2_2;
inputDeriv=[0,diff(obs_CO2_1)];
testOcc1=rawData(1:timeFrame,1);
testOcc1=testOcc1';

obs_CO22=rawData(1:timeFrame,5);
obs_CO22=obs_CO22';
[obs_CO2_2,cost] = tvd_mm(obs_CO22,100,100);
obs_CO2_2=obs_CO2_2';
obs_CO2_2 = smoothdata(obs_CO2_2,"sgolay",100);
% figure(1)
% clf
% hold on
% plot(obs_CO2)
% plot(obs_CO2_2)
obs_CO2_2=obs_CO2_2;
inputDeriv=[0,diff(obs_CO2_2)];
testOcc2=rawData(1:timeFrame,2);
testOcc2=testOcc2';
%^^^ SYNTHETIC DATASET


% %\/\/\/ MEETING ROOM DATASET
% rawData=table2array(readtable('MeetingRoom_Lounge_1May_17MayWithDerivatives.csv'));
% lastSaveSuffix="Meeting_properV_NV";
% timeFrame=size(rawData,1);
% obs_CO2=rawData(1:timeFrame,4);
% obs_CO2=obs_CO2';
% obs_CO2_2 = smoothdata(obs_CO2,"gaussian",60);
% figure(1)
% clf
% hold on
% plot(obs_CO2)
% plot(obs_CO2_2)
% obs_CO2=obs_CO2_2;
% inputDeriv=[0,diff(obs_CO2)];
% testOcc=rawData(1:timeFrame,21);
% testOcc=testOcc';
% %^^^ MEETING ROOM DATASET

% %\/\/\/ CONFERENCE ROOM DATASET
% rawData=table2array(readtable('ConferenceRoomWithDerivatives.csv'));
% lastSaveSuffix="Conference_properV_NV";
% timeFrame=size(rawData,1);
% obs_CO2=rawData(1:timeFrame,4);
% obs_CO2=obs_CO2';
% [obs_CO2_2,cost] = tvd_mm(obs_CO2,100,100);
% obs_CO2_2=obs_CO2_2';
% obs_CO2_2 = smoothdata(obs_CO2_2,"sgolay",10);
% figure(1)
% clf
% hold on
% plot(obs_CO2)
% plot(obs_CO2_2)
% obs_CO2=obs_CO2_2;
% inputDeriv=[0,diff(obs_CO2)];
% testOcc=rawData(1:timeFrame,15);
% testOcc=testOcc';
% %^^^ CONFERENCE ROOM DATASET


% %\/\/\/ ANIAF DATASET
% rawData=table2array(readtable('ANIAFWithDerivatives.csv'));
% lastSaveSuffix="ANIAF_properV_NV";
% timeFrame=size(rawData,1);
% obs_CO2=rawData(1:timeFrame,1);
% obs_CO2=obs_CO2';
% obs_CO2_2 = smoothdata(obs_CO2,"gaussian",60);
% figure(1)
% clf
% hold on
% plot(obs_CO2)
% plot(obs_CO2_2)
% obs_CO2=obs_CO2_2;
% inputDeriv=[0,diff(obs_CO2)];
% testOcc=rawData(1:timeFrame,4);
% testOcc=testOcc';
% plot(testOcc*100+300)
% %^^^ ANIAF DATASET

% %\/\/\/ D1_0cb815081d30 DATASET
% rawData=table2array(readtable('D1_0cb815081d30.csv'));
% lastSaveSuffix="D1_0cb815081d30_properV_NV";
% timeFrame=size(rawData,1);
% obs_CO2=rawData(1:timeFrame,1);
% obs_CO2=obs_CO2';
% % [obs_CO2_2,cost] = tvd_mm(obs_CO2,100,100);
% % obs_CO2_2=obs_CO2_2';
% % obs_CO2_2 = smoothdata(obs_CO2_2,"sgolay",10);
% % figure(1)
% % clf
% % hold on
% % plot(obs_CO2)
% % plot(obs_CO2_2)
% % obs_CO2=obs_CO2_2;
% inputDeriv=[0,diff(obs_CO2)];
% testOcc=rawData(1:timeFrame,2);
% testOcc=testOcc';
% % testOcc(testOcc>2)=2;
% %^^^ D1_0cb815081d30 DATASET


numVars=5;% Even if number of variables are less than actual number of variables, there won't be any problem. It'll extend as it goes.
x(1,1)=0;
constraints(2,numVars)=0;

% The range for parameters to search for finding the optimal likelihood.
% First row is the lower bound
% Second row is the upper bound
constraints(1,1)=0.001;% p1
constraints(2,1)=0.02;
constraints(1,2)=0.001;% p2
constraints(2,2)=0.02;
constraints(1,3)=0.001;% p3
constraints(2,3)=0.02;
constraints(1,4)=0.001;% p4
constraints(2,4)=0.02;
constraints(1,5)=0.001;% p5
constraints(2,5)=0.02;
constraints(1,6)=0.001;% p6
constraints(2,6)=0.02;
% constraints(1,7)=0.01;% p_win1
% constraints(2,7)=0.2;
% constraints(1,8)=0.01;% p_win2
% constraints(2,8)=0.2;
constraints(1,7)=5;% half life 1
constraints(2,7)=300;
% constraints(1,10)=10;% half life 2
% constraints(2,10)=40;
constraints(1,8)=395;% unoccupied CO2 level
constraints(2,8)=405;
constraints(1,9)=700;% single occupied CO2 level
constraints(2,9)=780;
constraints(1,10)=10;% Sigma
constraints(2,10)=50;

constraints(1,11)=0.01;% d1
constraints(2,11)=0.02;
constraints(1,12)=0.01;% d2
constraints(2,12)=0.02;

constraints(1,13)=0.01;% room exchange
constraints(2,13)=0.85;

if isContinueLearning==0% Checks if we continue from last best solution or start from random solution
    for i=1:size(constraints,2)
        x(1,i)=constraints(1,i)+rand(1,1)*(constraints(2,i)-constraints(1,i));% Make random numbers between lower bound and upper bound.
    end
    % if x(1,9)<x(1,10)% Swap half life 1 and 2 if half life 1 is smaller than 2
    %     temp=x(1,9);
    %     x(1,9)=x(1,10);
    %     x(1,10)=temp;
    % end
    if x(1,8)>x(1,9)% Swap unoccupied and single occupancy CO2 parameters if single occupancy is smaller
        temp=x(1,8);
        x(1,8)=x(1,9);
        x(1,9)=temp;
    end
    % x=specialInit_Viterbi(x);% Start from special solution such that the transition matrices are diagonal
    bestX=x;% Just assume that the initial parameters are the best parameters we have so far
else
    load(strcat('lastSolution',lastSaveSuffix,'.mat'))% Load parameters from a file we saved before
    x=bestX;
end
minLoss=-inf;% The best loss value is initially -infinity
temperature=0.4;% This is used in Simulated Annealing Meta-Heuristic for optimizing the parameters
maxIter=2;% The number of iterations to run Simulated Annealing
occNoises(1,maxIter)=0;% Varaible to store noise in estimated occupancies
temps(1,maxIter)=0;% Variable to store the temperature in Simulated Annealing
for iter=1:maxIter
    if iter>1
        x=genSol(constraints,temperature,bestX);% Generate a new set of parameters based on temperature and previous best solution
    end
    % The parameters are separated from the vector of parameters in x
    pOcc=[x(1,1) x(1,2) x(1,3) x(1,4) x(1,5) x(1,6)];
    % pWin=[x(1,7) x(1,8)];
    hl=x(1,7);
    CO2_uocc=x(1,8);
    CO2_socc=x(1,9);
    sigma=x(1,10);

    pD=[x(1,11) x(1,12)];

    roomExchangeRate=x(1,13);

    % Run the viterbi algorithm here to get the best sequence given the fixed parameters
    [generatedCO2s,generatedOcc,generatedDoor,lossValue]=inference_Viterbi_NV(pOcc, pD, hl, roomExchangeRate, CO2_uocc, CO2_socc, sigma, obs_CO2_1, obs_CO2_2);
    occ_state=generatedOcc;
    %occ_state(occ_state>2)=2; % Convert the number of occupants into 3 states
    occStateSmoothed=[];
    occStateSmoothed(1,size(occ_state,2))=0;% Smooth the output by a window
    windowSize=4;% Window size
    for i=1:size(occ_state,2)
        if i>=windowSize+1 && i<=size(occ_state,2)-windowSize-1
            occStateSmoothed(1,i)=mode(occ_state(1,i-windowSize:i+windowSize));% Take mode of the values in the window
        else
            occStateSmoothed(1,i)=occ_state(1,i);
        end
    end

    occNoise=occNoiseMeasure(occStateSmoothed,6);% Calculate the noise of occupancy vector
    occNoises(1,iter)=occNoise;

    if minLoss<lossValue% If the Loss value current parameter set is better than best Loss value
        bestX=x;% Store best parameter set
        bestOccState=occ_state;% Store the best occupancy sequence
        % bestWinState=generatedVentillation;
        bestCO2s=generatedCO2s;% Store the best CO2 values
        minLoss=lossValue% Store the best Loss value
    end
    
    if temperature<0.01% If temperature is less than a small value, reset the temperature
        temperature=temperature+0.8;
        disp('Still Occ has not converged!!!')
    end
    if temperature>0.3% Use different temperature reduction schemes
        temperature=temperature*0.82;
    elseif temperature>0.1
        temperature=temperature*0.84;
    else
        temperature=temperature*0.86;
    end
    temps(1,iter)=temperature;

    % Show plots of noise and temperature
    % figure(4)
    % clf
    % hold on
    % plot(occNoises)
    % figure(5)
    % clf
    % hold on
    % plot(temps)
end

figure(4)
clf
hold on
plot(occNoises)
figure(5)
clf
hold on
plot(temps)

% Plot final Occupancy states and CO2
figure(2)
clf
hold on
plot(bestCO2s(1,:),'LineStyle','--')
plot(bestCO2s(2,:),'LineStyle','--')
plot(bestOccState(1,:)*100+300)
plot(bestOccState(2,:)*100+300)
% plot(bestWinState*100+550)
plot(obs_CO2_1,'LineStyle','--')
plot(obs_CO2_2,'LineStyle','--')
plot(testOcc1*120+320)
plot(testOcc2*120+320)

confusionmat(testOcc1,bestOccState(1,:))% Calculte confusion matrix
confusionmat(testOcc2,bestOccState(2,:))% Calculte confusion matrix

cfm=confusionmat(testOcc1,bestOccState(1,:))% Recalculate confusion matrix
acc=sum(diag(cfm))/sum(sum(cfm))% Report the accuracy

cfm=confusionmat(testOcc2,bestOccState(2,:))% Recalculate confusion matrix
acc=sum(diag(cfm))/sum(sum(cfm))% Report the accuracy

bestOccState(bestOccState>2)=2;
testOcc1(testOcc1>2)=2;
testOcc2(testOcc2>2)=2;

bestOccStateSmoothed(2,size(bestOccState,2))=0;% Smooth the final occupancy sequence
windowSize=4;
for i=1:size(bestOccState,2)
    if i>=windowSize+1 && i<=size(bestOccState,2)-windowSize-1
        bestOccStateSmoothed(1,i)=mode(bestOccState(1,i-windowSize:i+windowSize));
        bestOccStateSmoothed(2,i)=mode(bestOccState(2,i-windowSize:i+windowSize));
    else
        bestOccStateSmoothed(1,i)=bestOccState(1,i);
        bestOccStateSmoothed(2,i)=bestOccState(2,i);
    end
end

%plot(bestOccStateSmoothed(1,:)*130+340)% Plot smoothed occupancy states
%plot(bestOccStateSmoothed(2,:)*130+340)% Plot smoothed occupancy states
%legend('CO2 generated R1','CO2 generated R2','occ generated R1','occ generated R2','CO2 obs R1','CO2 obs R2','occ obs R1','occ obs R2')

plot(realDoor*20+260)
plot(generatedDoor*20+270)

legend('CO2 generated R1','CO2 generated R2','occ generated R1','occ generated R2','CO2 obs R1','CO2 obs R2','occ obs R1','occ obs R2','real Door','gen Door')
%legend('CO2 generated R1','CO2 generated R2','CO2 obs R1','CO2 obs R2')
cfmS=confusionmat(testOcc1,bestOccStateSmoothed(1,:))% Recalculate confusion matrix
accS=sum(diag(cfmS))/sum(sum(cfmS))% Report the accuracy

cfmS=confusionmat(testOcc2,bestOccStateSmoothed(2,:))% Recalculate confusion matrix
accS=sum(diag(cfmS))/sum(sum(cfmS))% Report the accuracy
save(strcat('lastSolution',lastSaveSuffix,'.mat'),"bestX")% Save the best parameter values on a file
x=bestX;

% The parameters are separated from the vector of parameters in x
pOcc=[x(1,1) x(1,2) x(1,3) x(1,4) x(1,5) x(1,6)];
% pWin=[x(1,7) x(1,8)];
hl=x(1,7);
CO2_uocc=x(1,8);
CO2_socc=x(1,9);
sigma=x(1,10);

pD=[x(1,11) x(1,12)];

roomExchangeRate=x(1,13);

% trueWin=rawData(:,2)';

% bestWinState(1,1)=1;
% for i=1:size(testOcc1,2)
%     if testOcc1(1,i)==0
%         trueWin(1,i)=1;
%         bestWinState(1,i)=1;
%     end
% end


% winSmoothed=[];
% winSmoothed(1,size(bestWinState,2))=0;
% windowSize=2;
% for i=1:size(bestWinState,2)
%     if i>=windowSize+1 && i<=size(bestWinState,2)-windowSize-1
%         winSmoothed(1,i)=mode(bestWinState(1,i-windowSize:i+windowSize))-1;
%     else
%         winSmoothed(1,i)=bestWinState(1,i)-1;
%     end
% end
% 
% figure(9)
% clf
% hold on
% plot(trueWin+0.1)
% plot(winSmoothed)
% ylim([-0.1 1.1])