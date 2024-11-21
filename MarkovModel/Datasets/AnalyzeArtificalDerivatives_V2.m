clc
clear
data=readtable('../artificialData_threeState.csv');
data=table2array(data);
deriv=diff(data(:,1));
data=data(2:size(data,1),:);
threshPosDeriv = 0.2
threshNegDeriv = -0.2
threshOcc1PosDeriv = 0.9
for i=1:size(data,1)
    sce(i)=getSCE(data(i,1),deriv(i,1),threshNegDeriv,threshPosDeriv,threshOcc1PosDeriv);
    if sce(i)==0 || sce(i)==1 || sce(i)==6 || sce(i)==10
        colors(i)=0;
    end
    if sce(i)==2 || sce(i)==3 || sce(i)==7
        colors(i)=1;
    end
    if sce(i) == 4 || sce(i) == 5 || sce(i) == 8 || sce(i) == 9
        colors(i)=2;
    end
end
figure(1)
clf
hold on
plot((data(:,1)/400)-1)
plot(deriv)
scatter(1:size(data,1),colors,[],colors)

function final_obsValIndexVal=getSCE(co2,deriv,threshNegDeriv,threshPosDeriv,threshOcc1PosDeriv)
threshAmbientAbovePercentage = 1.01 * (400 + (600 - 400) / 2);
threshOcc1AbovePercentage = 1.01 * (600 + (800 - 600) / 2);
sigmoid_deriv_G5 = 11+11 * round(-1 / (1 + exp(-20 * (deriv - (-threshOcc1PosDeriv)))));
sigmoid_deriv_G5_extra = 51+51 * round(-1 / (1 + exp(-20 * (deriv - (-threshOcc1PosDeriv)))));
sigmoid_deriv_G4 = -6+6 * round(1 / (1 + exp(-20 * (deriv - (threshNegDeriv)))));
sigmoid_deriv_G2 = 3 * round(-1 / (1 + exp(-20 * (deriv - (threshPosDeriv)))));
sigmoid_deriv_G1 = 7 * round(1 / (1 + exp(-20 * (deriv - (threshOcc1PosDeriv)))));
sigmoid_deriv_G1_extra = 41 * round(1 / (1 + exp(-20 * (deriv - (threshOcc1PosDeriv)))));
sigmoid_deriv=sigmoid_deriv_G1+sigmoid_deriv_G2+sigmoid_deriv_G4+sigmoid_deriv_G5;
sigmoid_deriv_extraG5=sigmoid_deriv_G1+sigmoid_deriv_G2+sigmoid_deriv_G4+sigmoid_deriv_G5_extra;
sigmoid_deriv_extraG5=round(sigmoid_deriv_extraG5/50);
sigmoid_deriv_extraG1=sigmoid_deriv_G1_extra+sigmoid_deriv_G2+sigmoid_deriv_G4+sigmoid_deriv_G5;
sigmoid_deriv_extraG1=round(sigmoid_deriv_extraG1/40);

sigmoid_co2_ambient = round(1+(1 / (1 + exp(-20 * (co2 - (threshAmbientAbovePercentage))))));
sigmoid_co2_occ = round((1 / (1 + exp(-20 * (co2 - (threshOcc1AbovePercentage))))));
sigmoid_co2 = sigmoid_co2_ambient+sigmoid_co2_occ;

final_obsValIndexVal = sigmoid_deriv+sigmoid_co2;
extraG5 = -1*sigmoid_deriv_extraG5*sigmoid_co2;
extraG1 = -1*sigmoid_deriv_extraG1*sigmoid_co2;
final_obsValIndexVal=final_obsValIndexVal+extraG5;
final_obsValIndexVal=final_obsValIndexVal+extraG1;
final_obsValIndexVal=final_obsValIndexVal+5;
end