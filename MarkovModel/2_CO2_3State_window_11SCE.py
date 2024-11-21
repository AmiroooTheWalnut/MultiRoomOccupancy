import numpyro
from numpyro.contrib.funsor.discrete import infer_discrete
from numpyro.contrib.funsor import config_enumerate
from numpyro.contrib.funsor.enum_messenger import markov
from threading import Thread
import multiprocessing

from numpyro.infer import init_to_feasible, init_to_median
from scipy import signal
import mtalg
from CustomInference import *
from DataHandler import *

smoothWindowLengthDeriv = 201
smoothWindowLengthInference = 31
isShowFigures=True

def fitted_transition(amb_co2, occ_co2, ambient_d_co2,occ1_d_co2,p_unk_o,p_pos_o1,p_pos_o2,p_neg_o,halflife, p_trn, p_win_cl, p_win_op, sigma, unique_name=False, threadIndex=-1):
    p_win=jnp.array([[1 - p_win_op, p_win_op],
                           [p_win_cl, 1 - p_win_cl]])

    # decay_d = 1.0 - jnp.exp2(- 1.0 / halflife_d)

    # co2_eq = jnp.array([amb_co2, occ_co2[0], occ_co2[1]])
    co2_eq_window = jnp.array([[amb_co2, occ_co2[0][0], occ_co2[0][1]],
                               [amb_co2, occ_co2[1][0], occ_co2[1][1]]])

    co2_d_p = jnp.array([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0,0.0,1.0],  # ambient
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0,0.0,1.0,0.0,0.0,0.0],  # occ1
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0,0.0,0.0,1.0,1.0,0.0],  # occ2
    ])

    def transition(state, obs_co2_t):
        t, occ_prev, win_prev, co2_prev,rng_key = state
        # now = datetime.datetime.now()
        # seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        # rng_key_unform = jax.random.PRNGKey(seed)
        suffix = f"_{t}" if unique_name else ""
        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            occ_t = numpyro.sample(
                "occ" + suffix + threadSuffix,
                dist.CategoricalProbs(p_trn[occ_prev]),
                infer={'enumerate': 'parallel'}, rng_key=rng_key
            )
            # win_t = numpyro.sample(
            #     "win" + suffix + threadSuffix,
            #     dist.CategoricalProbs(p_win[win_prev]),
            #     infer={'enumerate': 'parallel'}, rng_key=rng_key
            # )

            # win_t = 0
            # win_t_cl_p = numpyro.sample("win_cl" + suffix + threadSuffix,
            #                        dist.Uniform(0,p_win_cl))
            # win_t_op_p = numpyro.sample("win_op" + suffix + threadSuffix,
            #                             dist.Uniform(0, p_win_op))
            rVal=np.random.rand(1, 1).squeeze()
            # print("rVal: {}".format(rVal))
            # rVal=mtalg.random.random()
            # rVal = jax.random.uniform(rng_key_unform,minval=0, maxval=1)

            # rVal = numpyro.sample("rVal" + suffix, dist.Uniform(0, 1))

            # rVal = numpyro.sample(
            #     "rVal" + suffix + threadSuffix,
            #     dist.Uniform(0,1), rng_key=rng_key
            # )
            winTO=(1 + round(-1 / (1 + jnp.exp(20 * (rVal - p_win_op))))) * 2
            winTC = (-1+round(1/(1+jnp.exp(-20*(rVal-p_win_cl)))))*2
            win_t=jnp.array(jnp.heaviside(win_prev + winTO + winTC,0)).astype('int64')



            # if np.random.rand(1,1)>win_t_op_p and np.random.rand(1,1)<win_t_cl_p:
            #     win_t = 1
            # if np.random.rand(1,1)>win_t_cl_p and np.random.rand(1,1)<win_t_op_p:
            #     win_t = 0

        else:
            occ_t = numpyro.sample(
                "occ" + suffix,
                dist.CategoricalProbs(p_trn[occ_prev]),
                infer={'enumerate': 'parallel'}, rng_key=rng_key
            )
            # win_t = numpyro.sample(
            #     "win" + suffix,
            #     dist.CategoricalProbs(p_win[win_prev]),
            #     infer={'enumerate': 'parallel'}, rng_key=rng_key
            # )

            # win_t = 0

            # win_t_cl_p = numpyro.sample("win_cl" + suffix,
            #                             dist.Uniform(0, p_win_cl))
            # win_t_op_p = numpyro.sample("win_op" + suffix,
            #                             dist.Uniform(0, p_win_op))
            rVal = np.random.rand(1, 1).squeeze()
            # print("rVal: {}".format(rVal))
            # rVal = mtalg.random.random()
            # rVal = jax.random.uniform(rng_key_unform,minval=0,maxval=1)

            # rVal = numpyro.sample("rVal" + suffix,dist.Uniform(0, 1))


            # rVal = numpyro.sample(
            #     "rVal" + suffix,
            #     dist.Uniform(0, 1), rng_key=rng_key
            # )
            winTO = (1 + jnp.round(-1 / (1 + jnp.exp(20 * (rVal - p_win_op))))) * 2
            winTC = (-1 + jnp.round(1 / (1 + jnp.exp(-20 * (rVal - p_win_cl))))) * 2
            win_t = jnp.array(jnp.heaviside(win_prev + winTO + winTC,0)).astype('int64')



            # if np.random.rand(1, 1) > win_t_op_p and np.random.rand(1, 1) < win_t_cl_p:
            #     win_t = 1
            # elif np.random.rand(1, 1) > win_t_cl_p and np.random.rand(1, 1) < win_t_op_p:
            #     win_t = 0
            # else:
            #     win_t = win_prev
            # win_t = numpyro.sample(
            #     "win" + suffix,
            #     dist.CategoricalProbs(p_win[win_prev]),
            #     infer={'enumerate': 'parallel'}, rng_key=rng_key
            # )

        # a = numpyro.sample('mu', dist.Categorical(logits=jnp.array([0.2,0.3,0.5])))
        #
        # print("a= {}".format(a))

        threshPosDeriv = ambient_d_co2
        threshNegDeriv = -ambient_d_co2

        # threshPosDeriv = 0.15
        # threshNegDeriv = -0.15

        threshOcc1PosDeriv = occ1_d_co2

        # threshOcc1PosDeriv = 0.35

        # threshExtremePosDeriv = 1.2

        threshAmbientAbovePercentage = 1.001 * (amb_co2 + (occ_co2[win_t][0] - amb_co2) / 2)
        threshOcc1AbovePercentage = 1.001 * (occ_co2[win_t][0] + (occ_co2[win_t][1] - occ_co2[win_t][0]) / 2)

        # threshAmbientAbovePercentage = 684
        # threshOcc1AbovePercentage = 520

        sigmoid_deriv_G5 = 11 + 11 * round(-1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (-threshOcc1PosDeriv)))))
        sigmoid_deriv_G5_extra = 51 + 51 * round(-1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (-threshOcc1PosDeriv)))))
        sigmoid_deriv_G4 = -6 + 6 * round(1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (threshNegDeriv)))))
        sigmoid_deriv_G2 = 3 * round(-1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (threshPosDeriv)))))
        sigmoid_deriv_G1 = 7 * round(1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (threshOcc1PosDeriv)))))
        sigmoid_deriv_G1_extra = 41 * round(1 / (1 + jnp.exp(-20 * (obs_co2_t[1] - (threshOcc1PosDeriv)))))
        sigmoid_deriv = sigmoid_deriv_G1 + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5
        sigmoid_deriv_extraG5 = sigmoid_deriv_G1 + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5_extra
        sigmoid_deriv_extraG5 = round(sigmoid_deriv_extraG5 / 50)
        sigmoid_deriv_extraG1 = sigmoid_deriv_G1_extra + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5
        sigmoid_deriv_extraG1 = round(sigmoid_deriv_extraG1 / 40)

        sigmoid_co2_ambient = round(1 + (1 / (1 + jnp.exp(-20 * (obs_co2_t[0] - (threshAmbientAbovePercentage))))))
        sigmoid_co2_occ = round((1 / (1 + jnp.exp(-20 * (obs_co2_t[0] - (threshOcc1AbovePercentage))))))
        sigmoid_co2 = sigmoid_co2_ambient + sigmoid_co2_occ

        final_obsValIndexVal = sigmoid_deriv + sigmoid_co2
        extraG5 = -1 * sigmoid_deriv_extraG5 * sigmoid_co2
        extraG1 = -1 * sigmoid_deriv_extraG1 * sigmoid_co2
        final_obsValIndexVal = final_obsValIndexVal + extraG5
        final_obsValIndexVal = final_obsValIndexVal + extraG1
        final_obsValIndexVal = final_obsValIndexVal + 5
        final_obsValIndex = jnp.array([final_obsValIndexVal]).astype('int64').squeeze()

        decay = 1.0 - jnp.exp2(- 1.0 / halflife[win_t])

        co2_t_mean = co2_prev + decay * (co2_eq_window[win_t][occ_t] - co2_prev)
        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            co2_t = numpyro.sample("co2" + suffix+threadSuffix,
                                   dist.Normal(co2_t_mean, sigma), obs=obs_co2_t[0])
            co2_d_t = numpyro.sample("co2_d" + suffix+threadSuffix,
                                     dist.Categorical(probs=co2_d_p[occ_t]), obs=final_obsValIndex)
        else:
            co2_t = numpyro.sample("co2" + suffix,
                                   dist.Normal(co2_t_mean, sigma), obs=obs_co2_t[0])
            co2_d_t = numpyro.sample("co2_d" + suffix,
                                     dist.Categorical(probs=co2_d_p[occ_t]), obs=final_obsValIndex)
        # co2_d_t=0.0

        return (t + 1, occ_t, win_t, co2_t,rng_key), None

    return transition

def fitted_model(obs_co2, isNeedPRNG, threadIndex=-1,bestSamples=None,bestSampleIndex=-1):
    amb_co2L = 395
    amb_co2H = 405

    # occ_co2L_cl = 590
    # occ_co2_1L_cl = 790
    # occ_co2H_cl = 615
    # occ_co2_1H_cl = 815
    #
    # occ_co2L_op = 490
    # occ_co2_1L_op = 690
    # occ_co2H_op = 510
    # occ_co2_1H_op = 710
    #
    # halflifeL_OP = 10
    # halflifeH_OP = 25
    #
    # halflifeL_CL = 80
    # halflifeH_CL = 120

    occ_co2L_cl = 598
    occ_co2_1L_cl = 797
    occ_co2H_cl = 610
    occ_co2_1H_cl = 808

    # occ_co2L_op = 700
    # occ_co2_1L_op = 800
    # occ_co2H_op = 1100
    # occ_co2_1H_op = 1500

    halflifeL_OP = 19
    halflifeH_OP = 21

    halflifeL_CL = 58
    halflifeH_CL = 62


    ambient_d_co2L=3
    ambient_d_co2H=18
    occ1_d_co2L=14
    occ1_d_co2H=100
    # We impose weak priors
    if isNeedPRNG == True:
        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)
        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            amb_co2 = numpyro.sample("amb_co2" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(amb_co2, occ_co2H_cl), rng_key=rng_key)  # ORIG
            occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(occ_co2L_cl, occ_co2H_cl), rng_key=rng_key)
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl), rng_key=rng_key)  # ORIG
            occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1L_cl, occ_co2_1H_cl), rng_key=rng_key)
            # amb_co2_op = numpyro.sample("amb_co2_op" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_op = numpyro.sample("occ_co2_op" + threadSuffix, dist.Uniform(amb_co2_op, occ_co2H_op), rng_key=rng_key)  # ORIG
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op" + threadSuffix, dist.Uniform(occ_co2_1_op, occ_co2_1H_op), rng_key=rng_key)  # ORIG

            # occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(occ_co2L_cl, occ_co2H_cl), rng_key=rng_key)
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1L_cl, occ_co2_1H_cl), rng_key=rng_key)
            # amb_co2_op = numpyro.sample("amb_co2_op" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_op = numpyro.sample("occ_co2_op" + threadSuffix, dist.Uniform(occ_co2L_op, occ_co2H_op), rng_key=rng_key)
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op" + threadSuffix, dist.Uniform(occ_co2_1L_op, occ_co2_1H_op), rng_key=rng_key)

        else:
            amb_co2 = numpyro.sample("amb_co2", dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(amb_co2, occ_co2H_cl), rng_key=rng_key)  # ORIG
            occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(occ_co2L_cl, occ_co2H_cl), rng_key=rng_key)
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl", dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl), rng_key=rng_key)  # ORIG
            amb_co2_op = numpyro.sample("amb_co2_op", dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_op = numpyro.sample("occ_co2_op", dist.Uniform(amb_co2_op, occ_co2H_op), rng_key=rng_key)  # ORIG
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op", dist.Uniform(occ_co2_1_op, occ_co2_1H_op), rng_key=rng_key)  # ORIG

            # occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(amb_co2_cl, occ_co2H_cl), rng_key=rng_key)
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl", dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl), rng_key=rng_key)
            # amb_co2_op = numpyro.sample("amb_co2_op", dist.Uniform(amb_co2L, amb_co2H), rng_key=rng_key)
            # occ_co2_1_op = numpyro.sample("occ_co2_op", dist.Uniform(amb_co2_op, occ_co2H_op), rng_key=rng_key)
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op", dist.Uniform(occ_co2_1_op, occ_co2_1H_op), rng_key=rng_key)

        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            halflifeOp = numpyro.sample("halflifeOp" + threadSuffix, dist.Uniform(halflifeL_OP, halflifeH_OP), rng_key=rng_key)
            halflifeCl = numpyro.sample("halflifeCl" + threadSuffix, dist.Uniform(halflifeL_CL, halflifeH_CL), rng_key=rng_key)
            ambient_d_co2 = numpyro.sample("ambient_d_co2" + threadSuffix, dist.Uniform(ambient_d_co2L, ambient_d_co2H), rng_key=rng_key)
            # occ1_d_co2 = numpyro.sample("occ1_d_co2" + threadSuffix, dist.Uniform(ambient_d_co2, occ1_d_co2H), rng_key=rng_key)# ORIG
            occ1_d_co2 = numpyro.sample("occ1_d_co2" + threadSuffix, dist.Uniform(occ1_d_co2L, occ1_d_co2H), rng_key=rng_key)
            # unk_d_co2 = numpyro.sample("unk_d_co2" + threadSuffix, dist.Uniform(-0.5, 0.5), rng_key=rng_key)
            # pos_d_co2 = numpyro.sample("pos_d_co2" + threadSuffix, dist.Uniform(0, 5), rng_key=rng_key)
            # neg_d_co2 = numpyro.sample("neg_d_co2" + threadSuffix, dist.Uniform(-5, 0), rng_key=rng_key)
            p_pos_o1 = numpyro.sample("p_pos_o1" + threadSuffix, dist.Uniform(0.9, 1), rng_key=rng_key)
            p_pos_o2 = numpyro.sample("p_pos_o2" + threadSuffix, dist.Uniform(0.9, 1), rng_key=rng_key)
            p_unk_o = numpyro.sample("p_unk_o" + threadSuffix, dist.Uniform(0.9, 1), rng_key=rng_key)
            p_neg_o = numpyro.sample("p_neg_o" + threadSuffix, dist.Uniform(0, 0.1), rng_key=rng_key)
        else:
            halflifeOp = numpyro.sample("halflifeOp", dist.Uniform(halflifeL_OP, halflifeH_OP), rng_key=rng_key)
            halflifeCl = numpyro.sample("halflifeCl", dist.Uniform(halflifeL_CL, halflifeH_CL), rng_key=rng_key)
            ambient_d_co2 = numpyro.sample("ambient_d_co2", dist.Uniform(ambient_d_co2L, ambient_d_co2H), rng_key=rng_key)
            # occ1_d_co2 = numpyro.sample("occ1_d_co2", dist.Uniform(ambient_d_co2, occ1_d_co2H), rng_key=rng_key)# ORIG
            occ1_d_co2 = numpyro.sample("occ1_d_co2", dist.Uniform(occ1_d_co2L, occ1_d_co2H), rng_key=rng_key)
            # unk_d_co2 = numpyro.sample("unk_d_co2", dist.Uniform(-0.5, 0.5), rng_key=rng_key)
            # pos_d_co2 = numpyro.sample("pos_d_co2", dist.Uniform(0, 5), rng_key=rng_key)
            # neg_d_co2 = numpyro.sample("neg_d_co2", dist.Uniform(-5, 0), rng_key=rng_key)
            p_pos_o1 = numpyro.sample("p_pos_o1", dist.Uniform(0.9, 1), rng_key=rng_key)
            p_pos_o2 = numpyro.sample("p_pos_o2", dist.Uniform(0.9, 1), rng_key=rng_key)
            p_unk_o = numpyro.sample("p_unk_o", dist.Uniform(0.9, 1), rng_key=rng_key)
            p_neg_o = numpyro.sample("p_neg_o", dist.Uniform(0, 0.1), rng_key=rng_key)

        # amb_co2 = jnp.array([amb_co2_cl,amb_co2_op])
        q = (occ_co2_1_cl-amb_co2) / halflifeCl
        occ1Win = q * halflifeOp
        occ2Win = 2 * q * halflifeOp
        # CO2_EQ = jnp.array([[400.0, 600.0, 800.0], [400.0, 400.0 + occ1Win, 400.0 + occ2Win]])
        # print("open occ1 CO2 {} open occ2 CO2 {}".format(amb_co2 + occ1Win,amb_co2 + occ2Win))
        occ_co2 = jnp.array([[occ_co2_1_cl, occ_co2_2_cl], [amb_co2 + occ1Win, amb_co2 + occ2Win]])
        # occ_co2_op = [occ_co2_1_op, occ_co2_2_op]

        # if threadIndex>-1:
        #     threadSuffix=str(threadIndex)
        #     if bestSampleIndex == -1:
        #         halflife_d = numpyro.sample("halflife_d" + threadSuffix, dist.Uniform(-30, 30), rng_key=rng_key)
        #     else:
        #         halflife_d = numpyro.sample("halflife_d" + threadSuffix, dist.Uniform(bestSamples["halflife_d"][bestSampleIndex], bestSamples["halflife_d"][bestSampleIndex]+1), rng_key=rng_key)
        #
        # else:
        #     if bestSampleIndex == -1:
        #         halflife_d = numpyro.sample("halflife_d", dist.Uniform(-30, 30), rng_key=rng_key)
        #     else:
        #         halflife_d = numpyro.sample("halflife_d", dist.Uniform(bestSamples["halflife_d"][bestSampleIndex], bestSamples["halflife_d"][bestSampleIndex]+1), rng_key=rng_key)

    else:
        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            amb_co2 = numpyro.sample("amb_co2" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(amb_co2, occ_co2H_cl))  # ORIG
            occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(occ_co2L_cl, occ_co2H_cl))
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl))  # ORIG
            occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1L_cl, occ_co2_1H_cl))
            # amb_co2_op = numpyro.sample("amb_co2_op" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_op = numpyro.sample("occ_co2_op" + threadSuffix, dist.Uniform(amb_co2_op, occ_co2H_op))  # ORIG
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op" + threadSuffix, dist.Uniform(occ_co2_1_op, occ_co2_1H_op))  # ORIG

            # occ_co2_1_cl = numpyro.sample("occ_co2_cl" + threadSuffix, dist.Uniform(occ_co2L_cl, occ_co2H_cl))
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl" + threadSuffix, dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl))
            # amb_co2_op = numpyro.sample("amb_co2_op" + threadSuffix, dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_op = numpyro.sample("occ_co2_op" + threadSuffix, dist.Uniform(occ_co2L_op, occ_co2H_op))
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op" + threadSuffix, dist.Uniform(occ_co2_1_op, occ_co2_1H_op))

        else:
            amb_co2 = numpyro.sample("amb_co2", dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(amb_co2, occ_co2H_cl))  # ORIG
            occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(occ_co2L_cl, occ_co2H_cl))
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl", dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl))  # ORIG
            occ_co2_2_cl = numpyro.sample("occ_co2_2_cl", dist.Uniform(occ_co2_1L_cl, occ_co2_1H_cl))
            # amb_co2_op = numpyro.sample("amb_co2_op", dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_op = numpyro.sample("occ_co2_op", dist.Uniform(amb_co2_op, occ_co2H_op))  # ORIG
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op", dist.Uniform(occ_co2_1_op, occ_co2_1H_op))  # ORIG

            # occ_co2_1_cl = numpyro.sample("occ_co2_cl", dist.Uniform(occ_co2L_cl, occ_co2H_cl))
            # occ_co2_2_cl = numpyro.sample("occ_co2_2_cl", dist.Uniform(occ_co2_1_cl, occ_co2_1H_cl))
            # amb_co2_op = numpyro.sample("amb_co2_op", dist.Uniform(amb_co2L, amb_co2H))
            # occ_co2_1_op = numpyro.sample("occ_co2_op", dist.Uniform(occ_co2L_op, occ_co2H_op))
            # occ_co2_2_op = numpyro.sample("occ_co2_2_op", dist.Uniform(occ_co2_1_op, occ_co2_1H_op))

        # amb_co2 = jnp.array([amb_co2_cl, amb_co2_op])
        # occ_co2 = jnp.array([[occ_co2_1_cl, occ_co2_2_cl],[occ_co2_1_op, occ_co2_2_op]])
        # # occ_co2_op = [occ_co2_1_op, occ_co2_2_op]
        if threadIndex>-1:
            threadSuffix=str(threadIndex)
            halflifeOp = numpyro.sample("halflifeOp" + threadSuffix, dist.Uniform(halflifeL_OP, halflifeH_OP))
            halflifeCl = numpyro.sample("halflifeCl" + threadSuffix, dist.Uniform(halflifeL_CL, halflifeH_CL))
            ambient_d_co2 = numpyro.sample("ambient_d_co2" + threadSuffix, dist.Uniform(ambient_d_co2L, ambient_d_co2H))
            # occ1_d_co2 = numpyro.sample("occ1_d_co2" + threadSuffix, dist.Uniform(ambient_d_co2, occ1_d_co2H))# ORIG
            occ1_d_co2 = numpyro.sample("occ1_d_co2" + threadSuffix, dist.Uniform(occ1_d_co2L, occ1_d_co2H))
            # unk_d_co2 = numpyro.sample("unk_d_co2" + threadSuffix, dist.Uniform(-0.5, 0.5))
            # pos_d_co2 = numpyro.sample("pos_d_co2" + threadSuffix, dist.Uniform(0, 5))
            # neg_d_co2 = numpyro.sample("neg_d_co2" + threadSuffix, dist.Uniform(-5, 0))
            p_pos_o1 = numpyro.sample("p_pos_o1" + threadSuffix, dist.Uniform(0.9, 1))
            p_pos_o2 = numpyro.sample("p_pos_o2" + threadSuffix, dist.Uniform(0.9, 1))
            p_unk_o = numpyro.sample("p_unk_o" + threadSuffix, dist.Uniform(0.9, 1))
            p_neg_o = numpyro.sample("p_neg_o" + threadSuffix, dist.Uniform(0, 0.1))

        else:
            halflifeOp = numpyro.sample("halflifeOp", dist.Uniform(halflifeL_OP, halflifeH_OP))
            halflifeCl = numpyro.sample("halflifeCl", dist.Uniform(halflifeL_CL, halflifeH_CL))
            ambient_d_co2 = numpyro.sample("ambient_d_co2", dist.Uniform(ambient_d_co2L, ambient_d_co2H))
            # occ1_d_co2 = numpyro.sample("occ1_d_co2", dist.Uniform(ambient_d_co2, occ1_d_co2H))# ORIG
            occ1_d_co2 = numpyro.sample("occ1_d_co2", dist.Uniform(occ1_d_co2L, occ1_d_co2H))
            # unk_d_co2 = numpyro.sample("unk_d_co2", dist.Uniform(-0.5, 0.5))
            # pos_d_co2 = numpyro.sample("pos_d_co2", dist.Uniform(0, 5))
            # neg_d_co2 = numpyro.sample("neg_d_co2", dist.Uniform(-5, 0))
            p_pos_o1 = numpyro.sample("p_pos_o1", dist.Uniform(0.9, 1))
            p_pos_o2 = numpyro.sample("p_pos_o2", dist.Uniform(0.9, 1))
            p_unk_o = numpyro.sample("p_unk_o", dist.Uniform(0.9, 1))
            p_neg_o = numpyro.sample("p_neg_o", dist.Uniform(0, 0.1))

        # amb_co2 = jnp.array([amb_co2_cl,amb_co2_op])
        q = (occ_co2_1_cl - amb_co2) / halflifeCl
        occ1Win = q * halflifeOp
        occ2Win = 2 * q * halflifeOp
        # CO2_EQ = jnp.array([[400.0, 600.0, 800.0], [400.0, 400.0 + occ1Win, 400.0 + occ2Win]])
        # print("open occ1 CO2 {} open occ2 CO2 {}".format(amb_co2 + occ1Win, amb_co2 + occ2Win))
        occ_co2 = jnp.array([[occ_co2_1_cl, occ_co2_2_cl], [amb_co2 + occ1Win, amb_co2 + occ2Win]])
        # occ_co2_op = [occ_co2_1_op, occ_co2_2_op]

        # if threadIndex>-1:
        #     threadSuffix=str(threadIndex)
        #     if bestSampleIndex == -1:
        #         halflife_d = numpyro.sample("halflife_d" + threadSuffix, dist.Uniform(-30, 30))
        #     else:
        #         halflife_d = numpyro.sample("halflife_d" + threadSuffix, dist.Uniform(bestSamples["halflife_d"][bestSampleIndex], bestSamples["halflife_d"][bestSampleIndex]+1))
        # else:
        #     if bestSampleIndex == -1:
        #         halflife_d = numpyro.sample("halflife_d", dist.Uniform(-30, 30))
        #     else:
        #         halflife_d = numpyro.sample("halflife_d", dist.Uniform(bestSamples["halflife_d"][bestSampleIndex], bestSamples["halflife_d"][bestSampleIndex]+1))

    numStates = 3
    p_arr = []
    for i in range(numStates):
        p_arr_row = []
        sumRow = 0
        for j in range(numStates - 1):
            # val = numpyro.param("p_arr_{}_{}".format(i, j), np.zeros(1) + 0.15, constraint=dist.constraints.interval(0, 1 - sumRow))

            if isNeedPRNG == True:
                now = datetime.datetime.now()
                seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
                rng_key = jax.random.PRNGKey(seed)
                # if i==j:
                #     val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0.9, 1 - sumRow), rng_key=rng_key)
                # else:
                #     val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0, jnp.min(jnp.array([1 - sumRow,0.1]))), rng_key=rng_key)
                if threadIndex > -1:
                    threadSuffix = str(threadIndex)
                    val = numpyro.sample("p_arr_{}_{}".format(i, j) + threadSuffix, dist.Uniform(0, 1 - sumRow), rng_key=rng_key)

                else:
                    val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0, 1 - sumRow), rng_key=rng_key)

            else:
                # if i == j:
                #     val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0.9, 1 - sumRow))
                # else:
                #     val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0, jnp.min(jnp.array([1 - sumRow,0.1]))))
                if threadIndex > -1:
                    threadSuffix = str(threadIndex)
                    val = numpyro.sample("p_arr_{}_{}".format(i, j) + threadSuffix, dist.Uniform(0, 1 - sumRow))
                else:
                    val = numpyro.sample("p_arr_{}_{}".format(i, j), dist.Uniform(0, 1 - sumRow))

            # val = numpyro.param("p_arr_{}_{}".format(i, j), np.zeros(1) + ((1 - sumRow)/2), constraint=dist.constraints.interval(0, 1 - sumRow))
            sumRow = sumRow + val
            p_arr_row.append(val)
        p_arr_row.append(jnp.abs(1 - sumRow))
        p_arr_row = jnp.array(p_arr_row).flatten()
        p_arr.append(p_arr_row)
    p_arr = jnp.array(p_arr)

    if isNeedPRNG == True:
        if threadIndex > -1:
            threadSuffix = str(threadIndex)
            p_w_o = numpyro.sample("p_w_o" + threadSuffix, dist.Uniform(0, 1), rng_key=rng_key)
            p_w_c = numpyro.sample("p_w_c" + threadSuffix, dist.Uniform(0, 1), rng_key=rng_key)
        else:
            p_w_o = numpyro.sample("p_w_o", dist.Uniform(0, 1), rng_key=rng_key)
            p_w_c = numpyro.sample("p_w_c", dist.Uniform(0, 1), rng_key=rng_key)
    else:
        if threadIndex > -1:
            threadSuffix = str(threadIndex)
            p_w_o = numpyro.sample("p_w_o" + threadSuffix, dist.Uniform(0, 1))
            p_w_c = numpyro.sample("p_w_c" + threadSuffix, dist.Uniform(0, 1))
        else:
            p_w_o = numpyro.sample("p_w_o", dist.Uniform(0, 1))
            p_w_c = numpyro.sample("p_w_c", dist.Uniform(0, 1))

    # p_win = jnp.array([[1-p_w_o,p_w_o],
    #                    [p_w_c,1-p_w_c]])

    if isNeedPRNG == True:
        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)
        if threadIndex > -1:
            threadSuffix = str(threadIndex)
            sigma = numpyro.sample("sigma" + threadSuffix, dist.Uniform(1, 40), rng_key=rng_key)
            # sigma_d = numpyro.sample("sigma_d" + threadSuffix, dist.Uniform(-30, 30), rng_key=rng_key)

        else:
            sigma = numpyro.sample("sigma", dist.Uniform(1, 40), rng_key=rng_key)
            # sigma_d = numpyro.sample("sigma_d", dist.Uniform(-30, 30), rng_key=rng_key)


    else:
        if threadIndex > -1:
            threadSuffix = str(threadIndex)
            sigma = numpyro.sample("sigma" + threadSuffix, dist.Uniform(1, 40))
            # sigma_d = numpyro.sample("sigma_d" + threadSuffix, dist.Uniform(-30, 30))

        else:
            sigma = numpyro.sample("sigma", dist.Uniform(1, 40))
            # sigma_d = numpyro.sample("sigma_d", dist.Uniform(-30, 30))


    # sigma = numpyro.param("sigma", np.zeros(1) + 15, constraint=dist.constraints.interval(0, 30))

    occ_init = 0

    co2_init=obs_co2[0,0]
    co2_d_init=0

    win_init = 0

    halflife = jnp.array([halflifeCl, halflifeOp])

    transition = fitted_transition(amb_co2, occ_co2, ambient_d_co2, occ1_d_co2,p_unk_o,p_pos_o1,p_pos_o2,p_neg_o,halflife, p_arr, p_w_c, p_w_o, sigma, threadIndex=threadIndex)
    rng_key = None
    if isNeedPRNG == True:
        now = datetime.datetime.now()
        seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)
        rng_key = jax.random.PRNGKey(seed)
    scan(transition, (0, occ_init, win_init, co2_init, rng_key), obs_co2[1:])

# @profile
def run(train_data, train_labels, test_data, test_labels, retVals, window_status, windowIndex=-1, seed=2):
    obs_co2_t_smoothed = signal.savgol_filter(test_data, window_length=smoothWindowLengthDeriv, polyorder=2, mode="interp", deriv=0)

    # box = np.ones(smoothWindowLength) / smoothWindowLength
    # obs_co2_t_smoothed = np.convolve(test_data, box, mode='same')

    # obs_co2_t_smoothed = moving_average(test_data, n=smoothWindowLength)


    # obs_co2_t_smoothed = test_data

    obs_co2_d_t = np.diff(obs_co2_t_smoothed.flatten())
    test_data = test_data[1:]

    # plt.plot(obs_co2_d_t, label='deriv')

    obs_co2_d_t_b=np.zeros((test_data.shape[0]))

    for i in range(test_data.shape[0]):
        threshPos = 0.2
        threshNeg = -0.2
        obs_sigmoid_pos = np.round(1 / (1 + jnp.exp(-20 * (obs_co2_d_t[i] - (threshPos)))))
        obs_sigmoid_neg = -np.round(1 / (1 + jnp.exp(-(-20 * (obs_co2_d_t[i] - (threshNeg))))))
        obs_sigmoid = jnp.array([obs_sigmoid_pos + obs_sigmoid_neg + 1]).astype('int64')
        obs_co2_d_t_b[i]=obs_sigmoid

    if isShowFigures==True:
        plt.plot((test_data[:] / 400) - 1, label='CO2')
        plt.plot(window_status*2,label='window')
        plt.plot(obs_co2_d_t_b, label='deriv adjusted')
        plt.plot((obs_co2_t_smoothed[:] / 400) - 1, label='smoothed')
        plt.legend()
        plt.show()

    test_data = np.transpose(np.array([test_data, obs_co2_d_t]))

    obs_co2_sce = np.zeros((test_data.shape[0]))

    threshPosDeriv = 0.5
    threshNegDeriv = -0.5

    # threshPosDeriv = 0.2
    # threshNegDeriv = -0.2

    threshExtremePosDeriv = 1.4
    # threshAmbientAbovePercentage = 1.01 * (400+(600-400)/3)
    # threshOcc1AbovePercentage = 1.01 * 600

    # threshAmbientAbovePercentage = 1.01 * (400+(600-400)/2)
    # threshOcc1AbovePercentage = 1.01 * (600+(800-600)/8)

    threshAmbientAbovePercentage = 1.001 * (400 + (600 - 400) / 2)
    threshOcc1AbovePercentage = 1.001 * (600 + (800 - 600) / 2)

    threshOcc1PosDeriv = 3

    colors = []
    XVals = []
    for i in range(test_data.shape[0]):
        XVals.append(i)
        sigmoid_deriv_G5 = 11 + 11 * round(-1 / (1 + jnp.exp(-20 * (test_data[i,1] - (-threshOcc1PosDeriv)))))
        sigmoid_deriv_G5_extra = 51 + 51 * round(-1 / (1 + jnp.exp(-20 * (test_data[i,1] - (-threshOcc1PosDeriv)))))
        sigmoid_deriv_G4 = -6 + 6 * round(1 / (1 + jnp.exp(-20 * (test_data[i,1] - (threshNegDeriv)))))
        sigmoid_deriv_G2 = 3 * round(-1 / (1 + jnp.exp(-20 * (test_data[i,1] - (threshPosDeriv)))))
        sigmoid_deriv_G1 = 7 * round(1 / (1 + jnp.exp(-20 * (test_data[i,1] - (threshOcc1PosDeriv)))))
        sigmoid_deriv_G1_extra = 41 * round(1 / (1 + jnp.exp(-20 * (test_data[i,1] - (threshOcc1PosDeriv)))))
        sigmoid_deriv = sigmoid_deriv_G1 + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5
        sigmoid_deriv_extraG5 = sigmoid_deriv_G1 + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5_extra
        sigmoid_deriv_extraG5 = round(sigmoid_deriv_extraG5 / 50)
        sigmoid_deriv_extraG1 = sigmoid_deriv_G1_extra + sigmoid_deriv_G2 + sigmoid_deriv_G4 + sigmoid_deriv_G5
        sigmoid_deriv_extraG1 = round(sigmoid_deriv_extraG1 / 40)

        sigmoid_co2_ambient = round(1 + (1 / (1 + jnp.exp(-20 * (test_data[i,0] - (threshAmbientAbovePercentage))))))
        sigmoid_co2_occ = round((1 / (1 + jnp.exp(-20 * (test_data[i,0] - (threshOcc1AbovePercentage))))))
        sigmoid_co2 = sigmoid_co2_ambient + sigmoid_co2_occ

        final_obsValIndexVal = sigmoid_deriv + sigmoid_co2
        extraG5 = -1 * sigmoid_deriv_extraG5 * sigmoid_co2
        extraG1 = -1 * sigmoid_deriv_extraG1 * sigmoid_co2
        final_obsValIndexVal = final_obsValIndexVal + extraG5
        final_obsValIndexVal = final_obsValIndexVal + extraG1
        final_obsValIndex = final_obsValIndexVal + 5
        obs_co2_sce[i]=final_obsValIndex
        if final_obsValIndex==0 or final_obsValIndex==1 or final_obsValIndex==6 or final_obsValIndex==10:
            colors.append(0)
        if final_obsValIndex==2 or final_obsValIndex==3 or final_obsValIndex==7:
            colors.append(1)
        if final_obsValIndex == 4 or final_obsValIndex == 5 or final_obsValIndex == 8 or final_obsValIndex == 9:
            colors.append(2)

    # if isShowFigures == True:
    #     plt.plot((test_data[:, 0] / 400) - 1, label='CO2')
    #     plt.plot(window_status*2,label='window')
    #     plt.plot(obs_co2_d_t, label='deriv adjusted')
    #     plt.plot(test_labels, label='True labels')
    #     # plt.plot(obs_co2_sce, label='discrete scenarios')
    #     plt.scatter(np.array(XVals), colors, c=np.array(colors))
    #     plt.title('0:just unoccupied, 2:long unoccupied, 3:long 1 occupied, 1:started 1 occupied, 4:long multioccupied, 5:started multioccupied', wrap=True)
    #     plt.legend()
    #     plt.show()
    #
    #     plt.subplots(figsize=(15, 4))
    #     plt.plot((test_data[:, 0]) - 1, label='CO2')
    #     plt.plot(test_labels * 200 + 400, label='True occupancy')
    #     plt.xlabel("Time")
    #     plt.ylabel("CO2")
    #     plt.legend()
    #     plt.show()
    #
    #     plt.subplots(figsize=(15, 4))
    #     plt.plot(obs_co2_d_t)
    #     plt.xlabel("Time")
    #     plt.ylabel("CO2 derivative")
    #     # plt.legend()
    #     plt.show()

    # mcmc = numpyro.infer.MCMC(
    #     numpyro.infer.NUTS(fitted_model, step_size=0.001, adapt_step_size=True, target_accept_prob=0.75, dense_mass=True, max_tree_depth=10, forward_mode_differentiation=False, find_heuristic_step_size=True),
    #     num_warmup=40,
    #     num_samples=260,
    #     num_chains=1,
    # )

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.BarkerMH(fitted_model, step_size=0.00001, adapt_step_size=True, target_accept_prob=0.7, dense_mass=True, init_strategy=init_to_median(num_samples=50)),
        num_warmup=400,
        num_samples=2600,
        num_chains=1,
    )

    # internalKernel = numpyro.infer.NUTS(fitted_model, step_size=0.0001, adapt_step_size=True, target_accept_prob=0.7, dense_mass=True, init_strategy=init_to_median(num_samples=500))
    # kernel = numpyro.infer.HMCECS(internalKernel, num_blocks=10)
    # mcmc = numpyro.infer.MCMC(kernel, num_warmup=400, num_samples=8600)

    # kernel = numpyro.infer.SA(fitted_model, adapt_state_size=50, dense_mass=True, init_strategy=init_to_median(num_samples=300))
    # mcmc = numpyro.infer.MCMC(kernel, num_warmup=400, num_samples=9600)

    # mcmc = numpyro.infer.MCMC(
    #     numpyro.infer.HMC(fitted_model, step_size=0.0001, adapt_step_size=True, target_accept_prob=0.7, dense_mass=False),
    #     num_warmup=40,
    #     num_samples=860,
    #     num_chains=1,
    # )

    # kernel = numpyro.infer.DiscreteHMCGibbs(numpyro.infer.NUTS(fitted_model), modified=True)
    # kernel = numpyro.infer.MixedHMC(numpyro.infer.HMC(fitted_model, trajectory_length=1.2), num_discrete_updates=20)
    # kernel = numpyro.infer.BarkerMH(fitted_model, step_size=0.1,adapt_step_size=True,target_accept_prob=0.6,dense_mass=True)
    # mcmc = numpyro.infer.MCMC(
    #     kernel,
    #     num_warmup=400,
    #     num_samples=49600,
    #     num_chains=1,
    # )

    # mcmc = numpyro.infer.MCMC(
    #     numpyro.infer.HMC(fitted_model),
    #     num_warmup=150,
    #     num_samples=1550,
    #     num_chains=1,
    # )

    now = datetime.datetime.now()
    seed = now.day + now.minute + now.second + now.month + now.microsecond + randrange(1000)

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, obs_co2=test_data, isNeedPRNG=False)
    mcmc.print_summary()

    samples = mcmc.get_samples()

    start_time = time.time()

    meanState = serialInferenceDeriv(fitted_model, test_data, numTopCo2=5,numTopCo2d=5, isThreadIndexAvailable=True, isSaveDebug=False)

    inferred_occ_np=meanState
    # meanState[meanState < 0.5] = 0
    # meanState[(meanState > 0.5) & (meanState < 1.334)] = 1
    # meanState[meanState > 1.334] = 2
    # inferred_occAll = meanState
    # inferred_occ_np = np.array(inferred_occAll)
    # # occ_smoothed = moving_average(inferred_occ_np, n=15)
    # # occ_smoothed = np.concatenate((occ_smoothed, inferred_occ_np[-14:]))

    occ_smoothed=signal.savgol_filter(inferred_occ_np, window_length=smoothWindowLengthInference, polyorder=2, mode="interp", deriv=0)

    occ_smoothed[occ_smoothed < 0.5] = 0
    occ_smoothed[(occ_smoothed > 0.5) & (occ_smoothed < 1.334)] = 1
    occ_smoothed[occ_smoothed > 1.334] = 2

    # occ_smoothed2 = moving_average(occ_smoothed, n=21)
    # occ_smoothed = np.concatenate((occ_smoothed2, occ_smoothed[-7:]))
    # occ_smoothed = signal.savgol_filter(occ_smoothed, window_length=smoothWindowLengthInference, polyorder=7, mode="interp", deriv=0)
    #
    # occ_smoothed[occ_smoothed < 0.5] = 0
    # occ_smoothed[(occ_smoothed > 0.5) & (occ_smoothed < 1.334)] = 1
    # occ_smoothed[occ_smoothed > 1.334] = 2

    inferred_occ_np = occ_smoothed

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    test_labels = test_labels.flatten()

    if test_labels.shape[0] < inferred_occ_np.shape[0]:
        inferred_occ_np = inferred_occ_np[(inferred_occ_np.shape[0] - test_labels.shape[0]):]
    elif test_labels.shape[0] > inferred_occ_np.shape[0]:
        test_labels = test_labels[(test_labels.shape[0] - inferred_occ_np.shape[0]):]

    acc = (test_labels.shape[0] - sum(np.abs(inferred_occ_np - test_labels))) / (test_labels.shape[0])
    print("NumPyroMCMC accuracy: {}".format(acc))

    retVals.append([test_labels, inferred_occ_np, acc, None])

    return samples

if __name__ == "__main__":
    print("Jax version {}".format(jax.__version__))
    numpyro.enable_x64(use_x64=True)
    numpyro.set_platform('cpu')
    # numpyro.set_host_device_count(5)

    # dataType = 'ConferenceRoom'
    # cols = [10]
    # # cols = [7,8,10]
    # windowIndex = 13

    # dataType = 'MeetingRoom'
    # cols = [10]
    # # cols = [7,8,10]
    # windowIndex = 14

    dataType = 'Artificial'
    # cols = [0,1]
    cols = [0]
    windowIndex = 1

    dataHandler = DataHandler()

    manager = multiprocessing.Manager()
    retVals = manager.list()
    train_data, train_labels, test_data, test_labels = dataHandler.handleDataRequest(0.0, dataType, isWindowsAvailable=True)
    window_status=test_data[:, windowIndex]
    train_data = train_data[:, cols]
    test_data = test_data[:, cols]
    test_data = test_data.flatten()

    reviseNumStates(train_labels, test_labels, isBinaryOccupancy=False, is3ClassLabel=True)

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    # np.savetxt("artificialData_threeState_400HL.csv", np.array([test_data, test_labels]).transpose(), delimiter=",")

    if isShowFigures == True:
        plt.plot(test_data, label='CO2')
        # plt.plot(400+rawData[:, 1]*50,label='window')
        plt.plot(400 + test_labels * 100, label='occupancy')
        plt.plot(420 + window_status * 100, label='window')
        plt.legend()
        plt.xticks(np.arange(0, len(test_labels) + 1, 100))
        plt.grid(axis='x')
        plt.show()

    samples=run(train_data, train_labels, test_data, test_labels, retVals, window_status, windowIndex=windowIndex, seed=2)

    DataHandler.saveResults(retVals, dataType, extraMessage="2_CO2_3State_window", isSaveROC=False, isSaveDetailed=False)

    # obs_co2_t_smoothed = signal.savgol_filter(test_data, window_length=smoothWindowLengthInference, polyorder=7, mode="interp", deriv=0)
    obs_co2_t_smoothed = test_data
    obs_co2_d_t = np.diff(obs_co2_t_smoothed)
    obs_co2_d_t_b = np.zeros((test_data.shape[0]))

    for i in range(obs_co2_d_t.shape[0]):
        threshPos = np.array(samples['ambient_d_co2']).mean()
        threshNeg = -np.array(samples['ambient_d_co2']).mean()
        obs_sigmoid_pos = np.round(1 / (1 + jnp.exp(-5 * (obs_co2_d_t[i] - (threshPos)))))
        obs_sigmoid_neg = -np.round(1 / (1 + jnp.exp(-(-5 * (obs_co2_d_t[i] - (threshNeg))))))
        obs_sigmoid = jnp.array([obs_sigmoid_pos + obs_sigmoid_neg + 1]).astype('int64')
        obs_co2_d_t_b[i] = obs_sigmoid

    plt.plot(test_data / 400, label='CO2 signals')
    # plt.plot(obs_co2_d_t_b + 0.2, label='CO2 deriv')
    plt.plot(retVals[0][1][6:] + 0.1, label='Inferred labels')
    plt.plot(retVals[0][0], label='True labels')
    plt.legend()

    plt.show()
