function [outputCO2,outputOcc,outputDoor,bestLoss]=inference_Viterbi_NV(pOcc, pD, hl, roomExchangeRate, CO2_uocc, CO2_socc, sigma, obs_CO2_1, obs_CO2_2)
numStates=6;% Maximum number of occupants is # - 1

% Construct the transition matrix
arrMat=[1-pOcc(1,1)-pOcc(1,2),  pOcc(1,1),              pOcc(1,2);
    pOcc(1,3),              1-pOcc(1,3)-pOcc(1,4),  pOcc(1,4);
    pOcc(1,5),              pOcc(1,6),              1-pOcc(1,5)-pOcc(1,6)];
% winMat=[1-pWin(1,1),    pWin(1,1)
%     pWin(1,2),      1-pWin(1,2)];

doorMat=[1-pD(1,1),    pD(1,1)
         pD(1,2),      1-pD(1,2)];

% Ensure the transition matrix is valud
for i=1:size(arrMat,1)
    for j=1:size(arrMat,2)
        arrMat(i,j)=abs(arrMat(i,j));
    end
    rSum=sum(arrMat(i,:));
    arrMat(i,:)=arrMat(i,:)/rSum;
end

% Calculate the cimulative probability for each row of the transition matrix
stateProbCum(size(arrMat,1),size(arrMat,2))=0;
for r=1:size(arrMat,1)
    stateProbCum(r,1)=arrMat(r,1);
    for i=2:size(arrMat,2)
        stateProbCum(r,i)=stateProbCum(r,i-1)+arrMat(r,i);
    end
end

% winStateProbCum(size(winMat,1),size(winMat,2))=0;
% for r=1:size(winMat,1)
%     winStateProbCum(r,1)=winMat(r,1);
%     for i=2:size(winMat,2)
%         winStateProbCum(r,i)=winStateProbCum(r,i-1)+winMat(r,i);
%     end
% end

T=size(obs_CO2_1,2);% Sequence length

S=2*numStates*numStates;% Number of steps

P_states(S,T)=0;% Matrix for tracing back in Viterbi
F_values(S,T)=0;% Matrix for forward pass
F_values(1:S,1)=1;% The initial values for time step 1
Full_path(1,T)=0;% The final occupancy sequences
Full_path(1,1)=1;% Assuming that the first time step has 0 occupancy
y_1(S,T)=0;% The CO2 values for each state and time step
y_1(1:S,1)=CO2_uocc;% Assuming that the first time step has fixed CO2 value for unoccupied
y_2(S,T)=0;% The CO2 values for each state and time step
y_2(1:S,1)=CO2_uocc;% Assuming that the first time step has fixed CO2 value for unoccupied
for t=2:T
    % disp('!!!')
    for occVal1=1:numStates
    for occVal2=1:numStates
        for doorVal=1:2
        F_valuesTemp=[];
        F_valuesTemp(S,1)=0;% Initializing the stage values calculated from previous stage
        internalYValues=[];
        internalYValues(S,1)=0;% Initializing the CO2 values for each combination of stage recursion
        for s=1:S
            prevDoor=floor((s-1)/(numStates*numStates))+1;
            prevOcc1=floor((s-1)/numStates)+1;
            prevOcc2=mod(s-1,numStates)+1;
            % prevWin=floor((s-1)/numStates)+1;
            % q=(CO2_socc-CO2_uocc)/hl(1,1);
            % alpha_socc=q*hl(1,2);
            % d=1-exp(-1/(hl(winVal)));
            alpha_socc=(CO2_socc-CO2_uocc);
            % d=1-exp(-1/(hl));
            d=1-2^(-1/(hl));
            % if winVal==1
            % expectedCO2=CO2_uocc+(CO2_socc-CO2_uocc)*(occVal-1);
            % else
            expectedCO2_1=CO2_uocc+(alpha_socc)*(occVal1-1);
            expectedCO2_2=CO2_uocc+(alpha_socc)*(occVal2-1);
            % end
            if t>1
                val1=y_1(s,t-1);
                y_t_1_temp=val1+d*(expectedCO2_1-(val1));
                val2=y_2(s,t-1);
                y_t_2_temp=val2+d*(expectedCO2_2-(val2));
                if doorVal==1
                    y_t_1=y_t_1_temp+(y_t_2_temp-y_t_1_temp)*roomExchangeRate*0.01;
                else
                    y_t_1=y_t_1_temp+(y_t_2_temp-y_t_1_temp)*roomExchangeRate;
                end
                
                if doorVal==1
                    y_t_2=y_t_2_temp+(y_t_1_temp-y_t_2_temp)*roomExchangeRate*0.01;
                else
                    y_t_2=y_t_2_temp+(y_t_1_temp-y_t_2_temp)*roomExchangeRate;
                end
            else
                y_t_1=CO2_uocc;
                y_t_2=CO2_uocc;
            end
            if y_t_1<10 || y_t_2<10
                disp('debug!!!')
            end
            CO2s_val1=randn(1)*sigma+y_t_1;
            CO2s_val2=randn(1)*sigma+y_t_2;
            likelihoodCO2_1=exp(-0.5 * ((obs_CO2_1(1,t) - y_t_1)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
            likelihoodCO2_2=exp(-0.5 * ((obs_CO2_2(1,t) - y_t_2)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
            % likelihoodCO2=MyNormpdf(obs_CO2(1,t),y_t,sigma);% Slow pdf calculation

            occValD1=occVal1;
            if occValD1>3
                occValD1=3;
            end
            if prevOcc1>3
                prevOcc1=3;
            end
            likelihoodOcc1=arrMat(prevOcc1,occValD1);

            occValD2=occVal2;
            if occValD2>3
                occValD2=3;
            end
            if prevOcc2>3
                prevOcc2=3;
            end
            likelihoodOcc2=arrMat(prevOcc2,occValD2);

            
            likelihoodDoor=doorMat(prevDoor,doorVal);

            % winValD=winVal;
            % if winValD>3
            %     winValD=3;
            % end
            % likelihoodWin=winMat(prevWin,winValD);

            % likelihoodTotal=likelihoodCO2*likelihoodOcc*likelihoodWin;
            likelihoodTotal=(likelihoodCO2_1+likelihoodCO2_2)*likelihoodOcc1*likelihoodOcc2*likelihoodDoor+1;% Avoid numerical issues by adding 1
            % likelihoodTotal=likelihoodCO2+1;

            F_valuesTemp(s,1)=(log(likelihoodTotal))+((F_values(s,t-1)));% Adding previous stage to current stage

            internalYValues1(s,1)=y_t_1;
            internalYValues2(s,1)=y_t_2;
        end
        currState=(doorVal-1)*(numStates*numStates)+((occVal1-1)*numStates)+occVal2;
        
        [val,bestPrevState]=max(F_valuesTemp(:,1));% Select the maximum value and index
        F_values(currState,t)=val;% Store the likelihood value
        P_states(currState,t)=bestPrevState;% Store the path i.e. from which state we have achieved the current state
        
        y_1(currState,t)=internalYValues1(bestPrevState,1);% Select the best CO2 value for the next time step calculation
        y_2(currState,t)=internalYValues2(bestPrevState,1);% Select the best CO2 value for the next time step calculation
        end
    end
    end
end
[bestLoss,I]=max(F_values(:,T));% Start from the highest likely state at the end of the sequence
Full_path(1,T)=I;% Store the best state index
prevI=P_states(I,T);% Select the previous state
Full_path(1,T-1)=prevI;
for t=T-1:-1:2% Move backwards to store all states
    prevI=P_states(prevI,t);
    Full_path(1,t-1)=prevI;
end

% Convert states into occupancy values
outputOcc(1,T)=0;
outputDoor(1,T)=0;
outputCO2(1,T)=0;
outputCO2(1,1)=CO2_uocc;
outputCO2(2,1)=CO2_uocc;
for t=2:T
    % occVal=Full_path(1,t)-1;
    doorVal=floor((Full_path(1,t)-1)/(numStates*numStates))+1;
    outputOcc(1,t)=floor((Full_path(1,t)-(((doorVal-1)*(numStates*numStates)))-1)/numStates);
    outputOcc(2,t)=mod(Full_path(1,t)-1,numStates);
    outputDoor(1,t)=doorVal;
    outputCO2(1,t)=y_1(Full_path(1,t),t);
    outputCO2(2,t)=y_2(Full_path(1,t),t);
end
% disp('!!!')