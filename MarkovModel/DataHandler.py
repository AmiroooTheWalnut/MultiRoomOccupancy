import math

import jax.numpy as jnp

import pandas as pd
import numpy as np
import numpyro
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import accuracy_score
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os

class DataHandler:
    def generateArtificialData(self, isReportFullCO2, isWindowsAvailable, isThreeState=False, seed=6, num_timesteps=14000):
        if isThreeState==False:
            P_ARR = 0.020  # Arrival probability
            P_DEP = 0.025  # Departure probability
            P_TRN = jnp.array([[1 - P_ARR, P_ARR],
                           [P_DEP, 1 - P_DEP]])
        else:
            P_ARR = 0.00056  # Arrival probability of a single person
            P_ARR_2 = 0.0016  # Arrival probability of 2 person
            P_DEP = 0.0007  # Departure probability of a single person
            P_DEP_2 = 0.0005  # Departure probability of 2 person
            P_TRN = jnp.array([[1 - P_ARR - P_ARR_2, P_ARR, P_ARR_2],
                               [P_DEP, 1 - P_DEP - P_ARR, P_ARR],
                               [P_DEP_2, P_DEP, 1 - P_DEP_2 - P_DEP]])
        if isWindowsAvailable==True:
            P_WIN_OPEN = 0.002  # Window open probability
            P_WIN_CLOSE = 0.001  # Window close probability
            P_WIN_TRN = jnp.array([[1 - P_WIN_OPEN, P_WIN_OPEN],
                           [P_WIN_CLOSE, 1 - P_WIN_CLOSE]])

        CO2_DCY_HLF_STD = 10  # halflife standard deviation

        # How quickly the CO2 decays to equilibrium. We assume the decay rate fluctuates on every timestep
        # We use gamma distribution for a) positive axis support b) pronounced peak and c) exponential tails
        if isWindowsAvailable==True:
            CO2_DCY_HLF_MEAN_WIN=jnp.array([60,20])
            # CO2_DCY_HLF_DIST_cl = dist.Gamma(
            #     rate=CO2_DCY_HLF_MEAN_WIN[0] / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
            #     concentration=(CO2_DCY_HLF_MEAN_WIN[0] / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            # )
            # CO2_DCY_HLF_DIST_op = dist.Gamma(
            #     rate=CO2_DCY_HLF_MEAN_WIN[1] / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
            #     concentration=(CO2_DCY_HLF_MEAN_WIN[1] / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            # )
            # CO2_DCY_HLF_DIST = jnp.array([CO2_DCY_HLF_DIST_cl, CO2_DCY_HLF_DIST_op])
            # # Verify that we got the mean and variance right
            # assert np.isclose(CO2_DCY_HLF_DIST_cl.mean, CO2_DCY_HLF_MEAN_WIN[0])
            # assert np.isclose(CO2_DCY_HLF_DIST_cl.variance, CO2_DCY_HLF_STD ** 2)
            # assert np.isclose(CO2_DCY_HLF_DIST_op.mean, CO2_DCY_HLF_MEAN_WIN[1])
            # assert np.isclose(CO2_DCY_HLF_DIST_op.variance, CO2_DCY_HLF_STD ** 2)
        else:
            CO2_DCY_HLF_MEAN = 40.0  # Average halflife
            # CO2_DCY_HLF_DIST = dist.Gamma(
            #     rate=CO2_DCY_HLF_MEAN / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
            #     concentration=(CO2_DCY_HLF_MEAN / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            # )
            # # Verify that we got the mean and variance right
            # assert np.isclose(CO2_DCY_HLF_DIST.mean, CO2_DCY_HLF_MEAN)
            # assert np.isclose(CO2_DCY_HLF_DIST.variance, CO2_DCY_HLF_STD ** 2)

        # Equilibrium CO2 levels for different occupancy levels. We don't assume any randomness for now
        if isWindowsAvailable==True:
            if isThreeState==False:
                CO2_EQ = jnp.array([[400.0, 800.0],[400.0,600.0]])
            else:
                # CO2_EQ = jnp.array([[400.0, 600.0, 800.0], [400.0, 500.0, 700.0]])
                q = 200/CO2_DCY_HLF_MEAN_WIN[0]
                occ1Win = q*CO2_DCY_HLF_MEAN_WIN[1]
                occ2Win = 2*q * CO2_DCY_HLF_MEAN_WIN[1]
                CO2_EQ = jnp.array([[400.0, 600.0, 800.0], [400.0, 400.0+occ1Win, 400.0+occ2Win]])
            # print("!!!")
        else:
            if isThreeState==False:
                CO2_EQ = jnp.array([400.0, 800.0])
            else:
                CO2_EQ = jnp.array([400.0, 600.0, 800.0])

        # We assume that the sensor measures the true distribution with some noise
        # and then averages with the previous value to smoothen the response
        CO2_SENSOR_NOISE_DIST = dist.Normal(5, 10)
        CO2_SENSOR_SMOOTHING = 0.8

        def true_transition_window(state, t):
            occupancy, true_co2, observed_co2, window = state

            # Markov chain transition
            occupancy = numpyro.sample('occ',
                                       dist.CategoricalProbs(P_TRN[occupancy]),
                                       infer={'enumerate': 'parallel'})

            window = numpyro.sample('win',
                                       dist.CategoricalProbs(P_WIN_TRN[window]),
                                       infer={'enumerate': 'parallel'})

            # equilibritum CO2 at this occupancy level
            eq_co2 = CO2_EQ[window,occupancy]

            CO2_DCY_HLF_DIST = dist.Gamma(
                rate=CO2_DCY_HLF_MEAN_WIN[window] / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
                concentration=(CO2_DCY_HLF_MEAN_WIN[window] / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            )
            # CO2_DCY_HLF_DIST_op = dist.Gamma(
            #     rate=CO2_DCY_HLF_MEAN_WIN[1] / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
            #     concentration=(CO2_DCY_HLF_MEAN_WIN[1] / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            # )
            # CO2_DCY_HLF_DIST = jnp.array([CO2_DCY_HLF_DIST_cl, CO2_DCY_HLF_DIST_op])

            # decay during this time step
            halflife = numpyro.sample('halflife', CO2_DCY_HLF_DIST)
            decay = 1.0 - jnp.exp2(- 1.0 / halflife)
            true_co2 += decay * (eq_co2 - true_co2)

            # sensor model
            sensor_noise = numpyro.sample('noise', CO2_SENSOR_NOISE_DIST)
            observed_co2 = (CO2_SENSOR_SMOOTHING) * observed_co2 \
                           + (1 - CO2_SENSOR_SMOOTHING) * (true_co2 + sensor_noise)
            new_state = (occupancy, true_co2, observed_co2, window)

            # Our true transition model is fully observable
            return new_state, new_state

        def true_transition(state, t):
            occupancy, true_co2, observed_co2 = state

            # Markov chain transition
            occupancy = numpyro.sample('occ',
                                       dist.CategoricalProbs(P_TRN[occupancy]),
                                       infer={'enumerate': 'parallel'})

            # equilibritum CO2 at this occupancy level
            eq_co2 = CO2_EQ[occupancy]

            CO2_DCY_HLF_DIST = dist.Gamma(
                rate=CO2_DCY_HLF_MEAN / CO2_DCY_HLF_STD ** 2,  # called beta in wikipedia
                concentration=(CO2_DCY_HLF_MEAN / CO2_DCY_HLF_STD) ** 2  # called alpha in wikipedia
            )

            # decay during this time step
            halflife = numpyro.sample('halflife', CO2_DCY_HLF_DIST)
            decay = 1.0 - jnp.exp2(- 1.0 / halflife)
            true_co2 += decay * (eq_co2 - true_co2)

            # sensor model
            sensor_noise = numpyro.sample('noise', CO2_SENSOR_NOISE_DIST)
            observed_co2 = (CO2_SENSOR_SMOOTHING) * observed_co2 \
                           + (1 - CO2_SENSOR_SMOOTHING) * (true_co2 + sensor_noise)
            new_state = (occupancy, true_co2, observed_co2)

            # Our true transition model is fully observable
            return new_state, new_state

        with numpyro.handlers.seed(rng_seed=seed):
            if isWindowsAvailable == True:
                _, state = scan(true_transition_window,
                            (0, CO2_EQ[0,0], CO2_EQ[0,0], 0),
                            jnp.arange(num_timesteps))
            else:
                _, state = scan(true_transition,
                                (0, CO2_EQ[0], CO2_EQ[0]),
                                jnp.arange(num_timesteps))

        if isWindowsAvailable == True:
            occ, co2, obs_co2, window = state
        else:
            occ, co2, obs_co2 = state

        if isReportFullCO2==True:
            obs_co2 = np.transpose(np.array([co2,obs_co2]))
        else:
            obs_co2 = np.array(obs_co2)

        occ = np.array(occ)
        if isWindowsAvailable==True:
            win = np.array(window)
            generatedData = np.transpose(np.array((obs_co2, win, occ)))
        else:
            generatedData = np.transpose(np.array((obs_co2, occ)))


        return generatedData

    def rawDataRequest(self,request, isWindowsAvailable=True, isReportFullCO2=False, isRunFromNotebook=False, isFeatureSelection=False):
        if request == "Artificial":
            rawData = self.generateArtificialData(isReportFullCO2, isWindowsAvailable, isThreeState=True)
            # plt.plot(rawData[:, 0],label='CO2')
            # # plt.plot(400+rawData[:, 1]*50,label='window')
            # plt.plot(400+rawData[:, 1]*100,label='occupancy')
            # plt.legend()
            # plt.show()
        elif request =="MeetingRoom":
            if isRunFromNotebook==False:
                rawData = pd.read_csv('..' + os.sep + 'Datasets' + os.sep + 'Preprocessed_Meetingroomfrom1May2023to17May2023.csv')
                header = list(rawData.columns.values)
            else:
                rawData = pd.read_csv('..' + os.sep + 'Backend' + os.sep + 'Datasets' + os.sep + 'Preprocessed_Meetingroomfrom1May2023to17May2023.csv')
                header = list(rawData.columns.values)
            if isFeatureSelection==True:
                rawData = rawData.iloc[:, [3, 5, 7, 8, 9, 10, 13, 14,15]]
                # header=[header[i] for i in [5, 7, 8, 9, 10, 13, 14,15]]
                # print('!!')

        elif request == "ConferenceRoom":
            rawData = pd.read_csv('..' + os.sep + 'Datasets' + os.sep + 'Preprocessed_Conferenceroomfrom19June2023to30June2023.csv')
            if isFeatureSelection==True:
                rawData=rawData.iloc[:,[5,7,8,9,10,13,14]]
                header = list(rawData.columns.values)
                # header = [header[i] for i in [5, 7, 8, 9, 10, 13, 14, 15]]
                # print('!!')

        if not isinstance(rawData,np.ndarray):
            rawData = rawData.to_numpy()
        if 'header' in locals():
            return rawData, header
        else:
            return rawData
    def handleDataRequestVal(self, trainP, valP, request, isReportFullCO2=False, isRunFromNotebook=False):
        """
        This function is called by external models when they request data.
        :param trainP:
        :param valP:
        :param request:
        :param isReportFullCO2:
        :param isRunFromNotebook:
        :return:
        """
        rawData=self.rawDataRequest(request,isReportFullCO2,isRunFromNotebook)

        if type(rawData) is tuple:
            rawData=rawData[0]

        num_timesteps = rawData.shape[0]
        num_columns = rawData.shape[1]
        trainEndIndex = math.floor(num_timesteps * trainP)
        valStartIndex = trainEndIndex + 1
        valEndIndex = trainEndIndex + math.floor(num_timesteps * valP)
        testStartIndex = valEndIndex + 1
        train_dataRaw = rawData[0:trainEndIndex,0:num_columns-1]
        train_labelsRaw = rawData[0:trainEndIndex,num_columns-1:num_columns]
        val_dataRaw = rawData[valStartIndex:valEndIndex,0:num_columns-1]
        val_labelsRaw = rawData[valStartIndex:valEndIndex,num_columns-1:num_columns]
        test_dataRaw = rawData[testStartIndex:num_timesteps,0:num_columns-1]
        test_labelsRaw = rawData[testStartIndex:num_timesteps,num_columns-1:num_columns]
        data=(train_dataRaw, train_labelsRaw, val_dataRaw, val_labelsRaw, test_dataRaw, test_labelsRaw)

        return data

    def handleDataRequest(self, trainP, request, isWindowsAvailable=True, isArtificialReportFullCO2=False, isRunFromNotebook=False, isFeatureSelection=False):
        """
        This function is called by external models when they request data.
        :param trainP:
        :param request:
        :param isReportFullCO2:
        :param isRunFromNotebook:
        :return:
        """
        rawData=self.rawDataRequest(request,isWindowsAvailable,isArtificialReportFullCO2,isRunFromNotebook, isFeatureSelection=isFeatureSelection)

        if type(rawData) is tuple:
            rawData=rawData[0]

        num_timesteps = rawData.shape[0]
        num_columns = rawData.shape[1]
        trainEndIndex = math.floor(num_timesteps * trainP)
        testStartIndex = trainEndIndex + 1
        train_dataRaw = rawData[0:trainEndIndex,0:num_columns-1]
        train_labelsRaw = rawData[0:trainEndIndex,num_columns-1:num_columns]
        test_dataRaw = rawData[testStartIndex:num_timesteps,0:num_columns-1]
        test_labelsRaw = rawData[testStartIndex:num_timesteps,num_columns-1:num_columns]
        data=(train_dataRaw, train_labelsRaw, test_dataRaw, test_labelsRaw)

        return data

    def handleDataRequest10Fold(self, request, foldIndex, isArtificialReportFullCO2=False, isRunFromNotebook=False,isFeatureSelection=False):
        """
        This function is called by external models when they request data with 10-fold cross validation.
        :param request:
        :param foldIndex:
        :param isReportFullCO2:
        :param isRunFromNotebook:
        :return:
        """
        rawData=self.rawDataRequest(request,isArtificialReportFullCO2,isRunFromNotebook,isFeatureSelection)
        if type(rawData) == tuple:
            rawData = rawData[0]
        data = self.get10Fold(rawData, foldIndex)

        return data

    def get10Fold(self,data,foldIndex):
        """
        This function splits "data" with respect to test "foldIndex". The output is 8 chunks of training data, 1 chunk of validation data,
        and one chunk of test data. Validation chunk is immediately before the test chunk (for first fold, it is immediately after it).
        :param data:
        :param foldIndex:
        :return (XTrainRaw,YTrainRaw,XValRaw,YValRaw,XTestRaw,YTestRaw):
        """
        XTrainRaw = np.empty((0,data.shape[1]-1))
        YTrainRaw = np.empty((0,1))
        XTestRaw = np.empty((0,data.shape[1]-1))
        YTestRaw = np.empty((0,1))
        for f in range(1,11):
            if f == foldIndex:
                # print("|||")
                testStart = math.floor(data.shape[0] * ((f - 1) / 10))
                testEnd = math.floor(data.shape[0] * (f / 10))
                XTestRaw = np.append(XTestRaw, data[testStart: testEnd, 0:data.shape[1] - 1]).reshape((-1, data.shape[1] - 1))
                YTestRaw = np.append(YTestRaw, data[testStart: testEnd, data.shape[1] - 1])
                # print("testStart {} testEnd {}".format(testStart,testEnd))
                # print("^^^")
            else:
                # print("|||")
                trainStart = math.floor(data.shape[0] * ((f - 1) / 10))
                trainEnd = math.floor(data.shape[0] * (f / 10))
                XTrainRaw = np.append(XTrainRaw, data[trainStart: trainEnd, 0:data.shape[1] - 1]).reshape((-1, data.shape[1] - 1))
                YTrainRaw = np.append(YTrainRaw, data[trainStart: trainEnd, data.shape[1] - 1])
                # print("trainStart {} trainEnd {}".format(trainStart, trainEnd))
                # print("^^^")

        return (XTrainRaw,YTrainRaw,XTestRaw,YTestRaw)

    def get10FoldVal(self,data,foldIndex):
        """
        This function splits "data" with respect to test "foldIndex". The output is 8 chunks of training data, 1 chunk of validation data,
        and one chunk of test data. Validation chunk is immediately before the test chunk (for first fold, it is immediately after it).
        :param data:
        :param foldIndex:
        :return (XTrainRaw,YTrainRaw,XValRaw,YValRaw,XTestRaw,YTestRaw):
        """
        XTrainRaw = np.empty((0,data.shape[1]-1))
        YTrainRaw = np.empty((0,1))
        XValRaw = np.empty((0,data.shape[1]-1))
        YValRaw = np.empty((0,1))
        XTestRaw = np.empty((0,data.shape[1]-1))
        YTestRaw = np.empty((0,1))
        for f in range(1,11):
            if foldIndex > 1:
                if f == foldIndex:
                    # print("|||")
                    testStart = math.floor(data.shape[0] * ((f - 1) / 10))
                    testEnd = math.floor(data.shape[0] * (f / 10))
                    XTestRaw=np.append(XTestRaw, data.iloc[testStart: testEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                    YTestRaw=np.append(YTestRaw, data.iloc[testStart: testEnd,data.shape[1]-1].to_numpy())
                    # print("testStart {} testEnd {}".format(testStart,testEnd))
                    # print("^^^")
                else:
                    if f == foldIndex - 1:
                        # print("|||")
                        valStart=math.floor(data.shape[0]*((f-1)/10))
                        valEnd=math.floor(data.shape[0]*(f/10))
                        XValRaw=np.append(XValRaw,data.iloc[valStart: valEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                        YValRaw=np.append(YValRaw,data.iloc[valStart: valEnd,data.shape[1]-1].to_numpy())
                        # print("valStart {} valEnd {}".format(valStart, valEnd))
                        # print("^^^")
                    else:
                        # print("|||")
                        trainStart = math.floor(data.shape[0] * ((f - 1) / 10))
                        trainEnd=math.floor(data.shape[0]*(f/10))
                        XTrainRaw=np.append(XTrainRaw,data.iloc[trainStart: trainEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                        YTrainRaw=np.append(YTrainRaw,data.iloc[trainStart: trainEnd,data.shape[1]-1].to_numpy())
                        # print("trainStart {} trainEnd {}".format(trainStart, trainEnd))
                        # print("^^^")
            else:
                if f == foldIndex:
                    # print("|||")
                    testStart=math.floor(data.shape[0]*((f-1)/10))
                    testEnd=math.floor(data.shape[0]*(f/10))
                    XTestRaw=np.append(XTestRaw,data.iloc[testStart: testEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                    YTestRaw=np.append(YTestRaw,data.iloc[testStart: testEnd,data.shape[1]-1].to_numpy())
                    # print("testStart {} testEnd {}".format(testStart, testEnd))
                    # print("^^^")
                elif f==foldIndex+1:
                    # print("|||")
                    valStart=math.floor(data.shape[0]*((f)/10))
                    valEnd=math.floor(data.shape[0]*((f+1)/10))
                    XValRaw=np.append(XValRaw,data.iloc[valStart: valEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                    YValRaw=np.append(YValRaw,data.iloc[valStart: valEnd,data.shape[1]-1].to_numpy())
                    # print("valStart {} valEnd {}".format(valStart, valEnd))
                    # print("^^^")
                else:
                    # print("|||")
                    trainStart=math.floor(data.shape[0]*((f-1)/10))
                    trainEnd=math.floor(data.shape[0]*(f/10))
                    XTrainRaw=np.append(XTrainRaw,data.iloc[trainStart: trainEnd,0:data.shape[1]-1].to_numpy()).reshape((-1,data.shape[1]-1))
                    YTrainRaw=np.append(YTrainRaw,data.iloc[trainStart: trainEnd,data.shape[1]-1].to_numpy())
                    # print("trainStart {} trainEnd {}".format(trainStart, trainEnd))
                    # print("^^^")

        return (XTrainRaw,YTrainRaw,XValRaw,YValRaw,XTestRaw,YTestRaw)

    @staticmethod
    def saveResults(results, dataType, extraMessage="", isSaveROC=False, isSaveDetailed=True):
        trueLabels = np.empty(0)
        inferredLabels = np.empty(0)
        allProbs = np.empty(0)
        for i in range(len(results)):
            trueLabels = np.append(trueLabels, results[i][0])
            inferredLabels = np.append(inferredLabels, results[i][1])
            if results[i][3] is not None:
                allProbs = np.append(allProbs, results[i][3]).reshape((-1, results[i][3].shape[1]))

            trueLabels = np.absolute(np.rint(trueLabels))
            inferredLabels = np.absolute(np.rint(inferredLabels))
            if isSaveDetailed == True:
                plt.plot(results[i][0], label='True labels')
                plt.plot(results[i][1], label='Inferred labels')
                plt.title("Acc {}".format(results[i][2]))
                plt.legend()
                plt.savefig("{}_{}_fold{}.png".format(extraMessage, dataType, i))
                plt.close()
                cm = confusion_matrix(results[i][0], results[i][1])

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(0, cm.shape[0]))))
                disp.plot()
                plt.title("Acc {}".format(results[i][2]))
                plt.savefig("{}_{}_ConfusionMatrix_fold{}.png".format(extraMessage, dataType, i))

                plt.close()
                if isSaveROC == True:
                    if np.unique(results[i][0]).shape[0] > 1:
                        test_labels_revised = results[i][0]
                        probs = results[i][3]
                        if np.unique(test_labels_revised).shape[0] != probs.shape[1]:
                            tempProbs = np.zeros((probs.shape[0], np.unique(test_labels_revised).shape[0]))
                            for i in range(probs.shape[0]):
                                tempProbs[i, 0] = probs[i, 0]
                                tempProbs[i, 1] = 1 - probs[i, 0]

                            probs = tempProbs
                        skplt.metrics.plot_roc_curve(test_labels_revised, probs)
                        plt.savefig("{}_{}_ROC_fold{}.png".format(extraMessage, dataType, i))

                        plt.close()
                    else:
                        print("ROC curve does not exist because the data has only one class instances")

        # trueLabels=np.array(trueLabels).flatten()
        # inferredLabels=np.array(inferredLabels).flatten()
        plt.plot(trueLabels, label='True labels')
        plt.plot(inferredLabels, label='Inferred labels')
        plt.legend()
        plt.savefig("{}_{}_all.png".format(extraMessage, dataType))

        plt.close()

        cm = confusion_matrix(trueLabels, inferredLabels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        accTotal = accuracy_score(trueLabels, inferredLabels)
        plt.title("Acc total {}".format(accTotal))
        plt.savefig("{}_{}_ConfusionMatrix_all.png".format(extraMessage, dataType))

        plt.close()

        cm = confusion_matrix(trueLabels, inferredLabels, normalize='true')

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        accTotal = accuracy_score(trueLabels, inferredLabels)
        plt.title("Acc total {}".format(accTotal))
        plt.savefig("{}_{}_ConfusionMatrix_all_norm.png".format(extraMessage, dataType))

        plt.close()

        if isSaveROC == True:
            if np.unique(trueLabels).shape[0] > 1:
                test_labels_revised = trueLabels
                probs = allProbs
                if np.unique(test_labels_revised).shape[0] != probs.shape[1]:
                    tempProbs=np.zeros((probs.shape[0], np.unique(test_labels_revised).shape[0]))
                    for i in range(probs.shape[0]):
                        tempProbs[i,0]=probs[i,0]
                        tempProbs[i, 1] = 1-probs[i, 0]
                    # probs = np.append(probs, np.zeros((probs.shape[1], probs.shape[1])), axis=0)
                    # for i in range(probs.shape[1]):
                    #     test_labels_revised = np.append(test_labels_revised, np.array([i]))
                    probs = tempProbs

                skplt.metrics.plot_roc_curve(test_labels_revised, probs)
                plt.savefig("{}_{}_ROC_all.png".format(extraMessage, dataType))

                plt.close()
            else:
                print("ROC curve does not exist because the data has only one class instances")

def reviseNumStates(train_labels, test_labels, isBinaryOccupancy=False, is3ClassLabel=False):
    if isBinaryOccupancy == True:
        train_labels[train_labels != 0] = 1
        test_labels[test_labels != 0] = 1
        numStates=2
    else:
        if is3ClassLabel == True:
            for i in range(train_labels.shape[0]):
                if train_labels[i]>1:
                    train_labels[i]=2
            for i in range(test_labels.shape[0]):
                if test_labels[i]>1:
                    test_labels[i]=2
            numStates = 3
        else:
            numStates = np.max(np.array([np.unique(train_labels).shape[0],np.unique(test_labels).shape[0]]))

    return numStates

if __name__ == "__main__":
    dataType = 'Artificial'
    dataHandler = DataHandler()
    data=dataHandler.rawDataRequest(dataType,isWindowsAvailable=True)
    np.savetxt("artificialData_threeState.csv", data, delimiter=",")
    print("Saved artifical data")
