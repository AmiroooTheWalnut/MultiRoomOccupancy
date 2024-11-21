import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import time
from random import randrange
from jax import random
import jax
import pandas as pd
import numpyro
from numpyro.infer import Predictive
from scipy import stats
import numpy as np

def uniqueColumn(col):
    return np.unique(col,  return_counts=True)

def initMCMC(fitted_model):
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.BarkerMH(fitted_model, step_size=0.2, adapt_step_size=True, target_accept_prob=0.7, dense_mass=True),
        num_warmup=200,
        num_samples=6800,
        num_chains=1,
    )

def fullInference(numCPUs,numIters,fitted_model,test_data, numSamples, numTopSamples, bestSamples=None,bestSampleIndex=-1):
    executor = ThreadPoolExecutor(numCPUs)
    futures=[]
    for iter in range(numIters):
        future = executor.submit(fullInferenceIteration, fitted_model,test_data, numSamples, numTopSamples, iter, bestSamples, bestSampleIndex)
        futures.append(future)
    wait(futures)
    inferred_occAll=np.zeros((numTopSamples,test_data.shape[0]-1))
    for iter in range(numIters):
        inferred_occ=futures[iter].result()
        if iter == 0:
            inferred_occAll = inferred_occ
        else:
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ), axis=0)
    executor.shutdown(wait=False)
    return inferred_occAll


def fullInferenceIteration(fitted_model,test_data, numSamples, numTopSamples, iter, samples, idx):
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.BarkerMH(fitted_model, step_size=0.2, adapt_step_size=True, target_accept_prob=0.7, dense_mass=True),
        num_warmup=2,
        num_samples=68,
        num_chains=1,
    )

    now = datetime.datetime.now()
    seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, obs_co2=test_data, threadIndex=idx, isNeedPRNG=False, bestSamples=samples,bestSampleIndex=idx)
    mcmc.print_summary()

    # samples = mcmc.get_samples()


    predictive = Predictive(fitted_model, num_samples=numSamples, infer_discrete=False, parallel=True)

    now = datetime.datetime.now()
    seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
    rng_key = jax.random.PRNGKey(seed)

    rng_key, rng_key_ = random.split(rng_key)

    for key in samples:
        samples[key]=np.array(samples[key])

    inferred_occ = predictive(rng_key_, obs_co2=test_data, isNeedPRNG=True, threadIndex=iter, bestSamples=samples,bestSampleIndex=idx)

    predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, test_data, False, iter)
    # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
    threadSuffix = str(iter)
    sortedIndices = np.argsort(-np.array(predictiveLogLikelihood['co2'+threadSuffix]), axis=0)
    data = np.array(inferred_occ['occ'+threadSuffix])
    inferred_occ = data[sortedIndices[0:numTopSamples], np.arange(data.shape[1])]
    return inferred_occ


def serialInference(fitted_model,test_data,isThreadIndexAvailable=False):
    # NEW INFERENCE
    start_time = time.time()
    numIters = 3
    for iter in range(numIters):
        print("Full samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/4)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        inferred_occ = predictive(rng_key_, obs_co2=test_data, isNeedPRNG=False, threadIndex=0)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, test_data, False, 0)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        if isThreadIndexAvailable==True:
            threadSuffix = str(0)
            sortedIndicesCo2 = np.argsort(-np.array(predictiveLogLikelihood['co2' + threadSuffix]), axis=0)
            data = np.array(inferred_occ['occ' + threadSuffix])
        else:
            sortedIndicesCo2 = np.argsort(-np.array(predictiveLogLikelihood['co2']), axis=0)
            data = np.array(inferred_occ['occ'])

        numBestSamples = 5
        inferred_occ = data[sortedIndicesCo2[0:numBestSamples], np.arange(data.shape[1])]
        if iter == 0:
            inferred_occAll = inferred_occ
        else:
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ), axis=0)

    # inferred_occAll=fullInference(1,4,fitted_model,test_data,300,30)
    # inferred_occAll = jax.vmap(fullInference)(1,3,fitted_model,test_data,300,30)

    print("Full samples finished")
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    numIters = 0
    for iter in range(numIters):
        print("Random samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/4)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        sampleIndices = np.random.choice(range(test_data.shape[0] - 1), np.round(test_data.shape[0]/50).astype(np.int64), replace=False)

        sampledTest_data = test_data[sampleIndices]

        inferred_occ = predictive(rng_key_, obs_co2=sampledTest_data, isNeedPRNG=False)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, sampledTest_data, False)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        sortedIndicesCo2 = np.argsort(-np.array(predictiveLogLikelihood['co2']), axis=0)
        data = np.array(inferred_occ['occ'])
        numBestSamples = 5
        inferred_occ = data[sortedIndicesCo2[0:numBestSamples], np.arange(data.shape[1])]
        if 'inferred_occAll' in locals():
            inferred_occ_full = np.zeros((inferred_occ.shape[0], inferred_occAll.shape[1])) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ_full), axis=0)
        else:
            inferred_occ_full = np.zeros((inferred_occ.shape[0], test_data.shape[0]-1)) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = inferred_occ_full

    masked_arr = np.ma.masked_where(inferred_occAll == -1, inferred_occAll)
    meanState = np.array(np.ma.mean(masked_arr, axis=0))
    return meanState

def serialInferenceDeriv(fitted_model,test_data,numTopCo2,numTopCo2d,isThreadIndexAvailable=False, isSaveDebug=False):
    # NEW INFERENCE
    start_time = time.time()
    numIters = 2
    for iter in range(numIters):
        print("Full samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/8)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        inferred_occ = predictive(rng_key_, obs_co2=test_data, isNeedPRNG=False, threadIndex=0)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, test_data, False, 0)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        if isThreadIndexAvailable==True:
            threadSuffix = str(0)
            co2LL = np.squeeze(predictiveLogLikelihood['co2' + threadSuffix])
            co2_dLL = np.squeeze(predictiveLogLikelihood['co2_d' + threadSuffix])
            LL=np.concatenate((co2LL,co2_dLL))
            sorted_LL = np.sort(-np.array(LL), axis=0)
            sorted_co2LL=np.sort(-np.array(co2LL), axis=0)
            sorted_co2_dLL = np.sort(-np.array(co2_dLL), axis=0)
            sortedIndicesAll = np.argsort(-np.array(LL), axis=0)
            sortedIndicesCo2 = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2' + threadSuffix])), axis=0)
            sortedIndicesCo2_d = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2_d' + threadSuffix])), axis=0)
            data = np.array(inferred_occ['occ' + threadSuffix])
        else:
            co2LL = np.squeeze(predictiveLogLikelihood['co2'])
            co2_dLL = np.squeeze(predictiveLogLikelihood['co2_d'])
            LL = np.concatenate((co2LL, co2_dLL))
            sorted_LL = np.sort(-np.array(LL), axis=0)
            sorted_co2LL = np.sort(-np.array(co2LL), axis=0)
            sorted_co2_dLL = np.sort(-np.array(co2_dLL), axis=0)
            sortedIndicesAll = np.argsort(-np.array(LL), axis=0)
            sortedIndicesCo2 = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2'])), axis=0)
            sortedIndicesCo2_d = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2_d'])), axis=0)
            data = np.array(inferred_occ['occ'])

        # print("!!!")
        # for i in range(sortedIndices.shape[0]):
        #     for j in range(sortedIndices.shape[1]):
        #         if sortedIndices[i][j]>=co2LL.shape[0]:
        #             sortedIndices[i]=sortedIndices[i]-co2LL.shape[0]
        sortedIndices=sortedIndicesAll.copy()
        sortedIndices[sortedIndices >= co2LL.shape[0]] -= co2LL.shape[0]
        # inferred_occ = data[np.concatenate((sortedIndicesCo2[0:numTopCo2],sortedIndicesCo2_d[0:numTopCo2d])), np.arange(data.shape[1])]
        inferred_occCo2 = data[sortedIndicesCo2[0:numTopCo2], np.arange(data.shape[1])]
        inferred_occCo2d = data[sortedIndicesCo2_d[0:numTopCo2d], np.arange(data.shape[1])]
        inferred_occ = data[sortedIndices[0:numTopCo2], np.arange(data.shape[1])]
        if iter == 0:
            inferred_occAll = inferred_occ
            inferred_occCo2All = inferred_occCo2
            inferred_occCo2dAll = inferred_occCo2d
        else:
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ), axis=0)
            inferred_occCo2All = np.concatenate((inferred_occCo2All, inferred_occCo2), axis=0)
            inferred_occCo2dAll = np.concatenate((inferred_occCo2dAll, inferred_occCo2d), axis=0)

    if isSaveDebug == True:
        np.savetxt("test_data.csv", test_data, delimiter=",")

        np.savetxt("co2LL.csv", np.around(co2LL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sorted_co2LL.csv", np.around(sorted_co2LL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sortedIndicesCo2.csv", np.around(sortedIndicesCo2,decimals=2), delimiter=",",fmt='%1.2f')

        np.savetxt("co2_dLL.csv", np.around(co2_dLL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sorted_co2_dLL.csv", np.around(sorted_co2_dLL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sortedIndicesCo2_d.csv", np.around(sortedIndicesCo2_d,decimals=2), delimiter=",",fmt='%1.2f')

        np.savetxt("LL.csv", np.around(LL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sorted_LL.csv", np.around(sorted_LL,decimals=2), delimiter=",",fmt='%1.2f')
        np.savetxt("sortedIndicesAll.csv", np.around(sortedIndicesAll,decimals=0), delimiter=",",fmt='%1.1f')

        np.savetxt("sortedIndicesCo2.csv", np.around(sortedIndicesCo2, decimals=0), delimiter=",",fmt='%1.1f')
        np.savetxt("sortedIndicesCo2_d.csv", np.around(sortedIndicesCo2_d, decimals=0), delimiter=",",fmt='%1.1f')
        np.savetxt("sortedIndices.csv", np.around(sortedIndices,decimals=0), delimiter=",",fmt='%1.1f')

        np.savetxt("inferred_occCo2.csv", np.around(inferred_occCo2,decimals=0), delimiter=",",fmt='%1.1f')
        np.savetxt("inferred_occCo2d.csv", np.around(inferred_occCo2d,decimals=0), delimiter=",",fmt='%1.1f')
        np.savetxt("inferred_occ.csv", np.around(inferred_occ,decimals=0), delimiter=",",fmt='%1.1f')

    # inferred_occAll=fullInference(1,4,fitted_model,test_data,300,30)
    # inferred_occAll = jax.vmap(fullInference)(1,3,fitted_model,test_data,300,30)

    print("Full samples finished")
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    numIters = 0
    for iter in range(numIters):
        print("Random samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/4)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        sampleIndices = np.random.choice(range(test_data.shape[0] - 1), np.round(test_data.shape[0]/50).astype(np.int64), replace=False)

        sampledTest_data = test_data[sampleIndices]

        inferred_occ = predictive(rng_key_, obs_co2=sampledTest_data, isNeedPRNG=False)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, sampledTest_data, False)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        sortedIndicesCo2 = np.argsort(-np.array(predictiveLogLikelihood['co2']), axis=0)
        data = np.array(inferred_occ['occ'])
        numBestSamples = 5
        inferred_occ = data[sortedIndicesCo2[0:numBestSamples], np.arange(data.shape[1])]
        if 'inferred_occAll' in locals():
            inferred_occ_full = np.zeros((inferred_occ.shape[0], inferred_occAll.shape[1])) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ_full), axis=0)
        else:
            inferred_occ_full = np.zeros((inferred_occ.shape[0], test_data.shape[0]-1)) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = inferred_occ_full

    inferred_occAll=inferred_occAll.astype(np.float32)
    inferred_occAll[inferred_occAll == -1] = np.nan
    modes = stats.mode(inferred_occAll, axis=0, nan_policy='omit')
    inferred_occAll=inferred_occAll.astype(np.int32)

    meanState=modes.mode.flatten()

    # masked_arr = np.ma.masked_where(inferred_occAll == -1, inferred_occAll)
    # meanState = np.array(np.ma.mean(masked_arr, axis=0))
    return meanState

def serialInferenceDerivV2(fitted_model,test_data,numTopCo2,numTopCo2d,isThreadIndexAvailable=False, isSaveDebug=False):
    # NEW INFERENCE
    start_time = time.time()
    numIters = 1
    allLL=[]
    allco2LL=[]
    allco2_dLL=[]
    for iter in range(numIters):
        print("Full samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/3)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        inferred_occ = predictive(rng_key_, obs_co2=test_data, isNeedPRNG=False, threadIndex=0)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, test_data, False, 0)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        if isThreadIndexAvailable==True:
            threadSuffix = str(0)
            co2LL = np.squeeze(predictiveLogLikelihood['co2' + threadSuffix])
            co2_dLL = np.squeeze(predictiveLogLikelihood['co2_d' + threadSuffix])
            LL=np.concatenate((co2LL,co2_dLL))
            allco2LL.append(co2LL)
            allco2_dLL.append(co2_dLL)
            allLL.append(LL)
            sorted_LL = np.sort(-np.array(LL), axis=0)
            sorted_co2LL=np.sort(-np.array(co2LL), axis=0)
            sorted_co2_dLL = np.sort(-np.array(co2_dLL), axis=0)
            sortedIndicesAll = np.argsort(-np.array(LL), axis=0)
            sortedIndicesCo2 = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2' + threadSuffix])), axis=0)
            sortedIndicesCo2_d = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2_d' + threadSuffix])), axis=0)
            data = np.array(inferred_occ['occ' + threadSuffix])
        else:
            co2LL = np.squeeze(predictiveLogLikelihood['co2'])
            co2_dLL = np.squeeze(predictiveLogLikelihood['co2_d'])
            LL = np.concatenate((co2LL, co2_dLL))
            allco2LL.append(co2LL)
            allco2_dLL.append(co2_dLL)
            allLL.append(LL)
            sorted_LL = np.sort(-np.array(LL), axis=0)
            sorted_co2LL = np.sort(-np.array(co2LL), axis=0)
            sorted_co2_dLL = np.sort(-np.array(co2_dLL), axis=0)
            sortedIndicesAll = np.argsort(-np.array(LL), axis=0)
            sortedIndicesCo2 = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2'])), axis=0)
            sortedIndicesCo2_d = np.argsort(-np.array(np.squeeze(predictiveLogLikelihood['co2_d'])), axis=0)
            data = np.array(inferred_occ['occ'])

        # print("!!!")
        # for i in range(sortedIndices.shape[0]):
        #     for j in range(sortedIndices.shape[1]):
        #         if sortedIndices[i][j]>=co2LL.shape[0]:
        #             sortedIndices[i]=sortedIndices[i]-co2LL.shape[0]
        sortedIndices=sortedIndicesAll.copy()
        sortedIndices[sortedIndices >= co2LL.shape[0]] -= co2LL.shape[0]
        # inferred_occ = data[np.concatenate((sortedIndicesCo2[0:numTopCo2],sortedIndicesCo2_d[0:numTopCo2d])), np.arange(data.shape[1])]
        inferred_occCo2 = data[sortedIndicesCo2[0:numTopCo2], np.arange(data.shape[1])]
        inferred_occCo2d = data[sortedIndicesCo2_d[0:numTopCo2d], np.arange(data.shape[1])]
        inferred_occ = data[sortedIndices[0:numTopCo2], np.arange(data.shape[1])]
        if iter == 0:
            inferred_occAll = inferred_occ
            inferred_occCo2All = inferred_occCo2
            inferred_occCo2dAll = inferred_occCo2d
        else:
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ), axis=0)
            inferred_occCo2All = np.concatenate((inferred_occCo2All, inferred_occCo2), axis=0)
            inferred_occCo2dAll = np.concatenate((inferred_occCo2dAll, inferred_occCo2d), axis=0)


    # inferred_occAll=fullInference(1,4,fitted_model,test_data,300,30)
    # inferred_occAll = jax.vmap(fullInference)(1,3,fitted_model,test_data,300,30)

    print("Full samples finished")
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    numIters = 0
    for iter in range(numIters):
        print("Random samples iteration {}".format(iter))
        predictive = Predictive(fitted_model, num_samples=np.uint32(np.round(test_data.shape[0]/6)).item(), infer_discrete=False, parallel=True)

        # predictive.get_samples()

        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)

        rng_key, rng_key_ = random.split(rng_key)

        sampleIndices = np.random.choice(range(test_data.shape[0] - 1), np.round(test_data.shape[0]/50).astype(np.int64), replace=False)

        sampledTest_data = test_data[sampleIndices]

        inferred_occ = predictive(rng_key_, obs_co2=sampledTest_data, isNeedPRNG=False)

        predictiveLogLikelihood = numpyro.infer.util.log_likelihood(fitted_model, inferred_occ, sampledTest_data, False)
        # indxes = np.argmax(np.array(predictiveLogLikelihood['co2']),axis=0)
        sortedIndicesCo2 = np.argsort(-np.array(predictiveLogLikelihood['co2']), axis=0)
        data = np.array(inferred_occ['occ'])
        numBestSamples = 5
        inferred_occ = data[sortedIndicesCo2[0:numBestSamples], np.arange(data.shape[1])]
        if 'inferred_occAll' in locals():
            inferred_occ_full = np.zeros((inferred_occ.shape[0], inferred_occAll.shape[1])) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = np.concatenate((inferred_occAll, inferred_occ_full), axis=0)
        else:
            inferred_occ_full = np.zeros((inferred_occ.shape[0], test_data.shape[0]-1)) - 1
            inferred_occ_full[:, sampleIndices[1:]] = inferred_occ
            inferred_occAll = inferred_occ_full

    inferred_occAll=inferred_occAll.astype(np.float32)
    inferred_occAll[inferred_occAll == -1] = np.nan
    modes = stats.mode(inferred_occAll, axis=0, nan_policy='omit')
    inferred_occAll=inferred_occAll.astype(np.int32)

    meanState=modes.mode.flatten()

    # masked_arr = np.ma.masked_where(inferred_occAll == -1, inferred_occAll)
    # meanState = np.array(np.ma.mean(masked_arr, axis=0))
    return meanState

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
