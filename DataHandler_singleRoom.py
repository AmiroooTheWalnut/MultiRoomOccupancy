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
    def generateArtificialData(self, isReportFullCO2, isWindowsAvailable, isThreeState=False, seed=5, num_timesteps=25001, inputHL=None):
        if isThreeState==False:
            P_ARR = 0.00041  # Arrival probability
            P_DEP = 0.00021  # Departure probability
            P_TRN = jnp.array([[1 - P_ARR, P_ARR],
                           [P_DEP, 1 - P_DEP]])
        else:
            P_ARR = 0.00011  # Arrival probability of a single person
            P_ARR_2 = 0.0010  # Arrival probability of 2 person
            P_DEP = 0.002  # Departure probability of a single person
            P_DEP_2 = 0.00009  # Departure probability of 2 person
            P_TRN = jnp.array([[1 - P_ARR - P_ARR_2, P_ARR, P_ARR_2],
                               [P_DEP, 1 - P_DEP - P_ARR, P_ARR],
                               [P_DEP_2, P_DEP, 1 - P_DEP_2 - P_DEP]])
        if isWindowsAvailable==True:
            P_WIN_OPEN = 0.003  # Window open probability
            P_WIN_CLOSE = 0.001  # Window close probability
            P_WIN_TRN = jnp.array([[1 - P_WIN_OPEN, P_WIN_OPEN],
                           [P_WIN_CLOSE, 1 - P_WIN_CLOSE]])

        CO2_DCY_HLF_STD = 10  # halflife standard deviation

        # How quickly the CO2 decays to equilibrium. We assume the decay rate fluctuates on every timestep
        # We use gamma distribution for a) positive axis support b) pronounced peak and c) exponential tails
        if isWindowsAvailable==True:
            if inputHL is None:
                CO2_DCY_HLF_MEAN_WIN=jnp.array([90,70])
            else:
                CO2_DCY_HLF_MEAN_WIN = jnp.array(inputHL)
        else:
            if inputHL is None:
                CO2_DCY_HLF_MEAN = 100.0  # Average halflife
            else:
                CO2_DCY_HLF_MEAN = inputHL  # Average halflife

        # Equilibrium CO2 levels for different occupancy levels. We don't assume any randomness for now
        if isWindowsAvailable==True:
            if isThreeState==False:
                CO2_EQ = jnp.array([[400.0, 800.0],[400.0,600.0]])
            else:
                q = 200/CO2_DCY_HLF_MEAN_WIN[0]
                occ1Win = q*CO2_DCY_HLF_MEAN_WIN[1]
                occ2Win = 2*q * CO2_DCY_HLF_MEAN_WIN[1]
                CO2_EQ = jnp.array([[400.0, 600.0, 800.0], [400.0, 400.0+occ1Win, 400.0+occ2Win]])
        else:
            if isThreeState==False:
                CO2_EQ = jnp.array([400.0, 800.0])
            else:
                CO2_EQ = jnp.array([400.0, 600.0, 800.0])

        # We assume that the sensor measures the true distribution with some noise
        # and then averages with the previous value to smoothen the response
        CO2_SENSOR_NOISE_DIST = dist.Normal(0, 20)
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

            # decay during this time step
            halflife = numpyro.sample('halflife', CO2_DCY_HLF_DIST)
            decay = 1.0 - jnp.exp2(- 1.0 / halflife)
            true_co2 += decay * (eq_co2 - true_co2)

            # sensor model
            sensor_noise = numpyro.sample('noise', CO2_SENSOR_NOISE_DIST)
            observed_co2 = (CO2_SENSOR_SMOOTHING) * observed_co2 + (1 - CO2_SENSOR_SMOOTHING) * (true_co2 + sensor_noise)
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
    dataHandler = DataHandler()
    rawData = dataHandler.generateArtificialData(isReportFullCO2=False, isWindowsAvailable=True, isThreeState=True,inputHL=None)

    np.savetxt("syntheticData.csv", rawData, delimiter=",")
    print("Saved artifical data")
