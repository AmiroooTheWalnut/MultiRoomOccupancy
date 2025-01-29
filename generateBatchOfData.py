import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def genData(hlInput,aeInput):
    print(hlInput)
    print(aeInput)
    np.random.seed(3)

    total_obs_time = 24  # hours

    possible_states = np.array([0, 1, 2, 3])

    hl_value = hlInput

    hl_value_str = str(hl_value)
    hl_value_split = hl_value_str.split(".")

    CO2_tres_1 = hl_value  # time resolution of CO2 sensor, min
    CO2_tres_2 = hl_value  # time resolution of CO2 sensor, min

    # Calculating CO2 emission rate, q

    C_co2 = 0.04  # volume concentraiton of CO2 in exhaled air
    Vt = 0.5  # L/breath, tidal voulume
    Rt = 15  # breaths/min, breathing rate

    V_room = 60000  # L, 3x4x5 m3 room volume

    q = Rt * Vt * C_co2 / V_room * 1e6  # ppm/min, CO2 emission rate
    print(f"q = {q} ppm/min")

    q = int(q)  # ppm/min
    A_value = 0.015  # 1/min, insulation characteristics of the room
    # (I've set A manually so that the equilibrium CO2 concentrations look natural; i.e. 730 ppm for 1 occupant, 1050 ppm for 2 occupants etc.)
    # This is the main parameter to play with to get different eq value.
    C_out = 400  # ppm

    room_exchange_rate = aeInput  # air exchange between rooms

    room_exchange_rate_str = str(room_exchange_rate)
    room_exchange_rate_split = room_exchange_rate_str.split(".")

    signal_noise_std = 10  # ppm
    experiment_noise_std = 1  # ppm (See C_N_with_noise function)

    observation_time = 5760  # minutes

    door_states = np.zeros(observation_time)
    num_door_changes = np.random.randint(7, 12)

    def C_N_Tplus1(C_T1, C_T2, N, D_S, RE, hl, experiment_noise_std=experiment_noise_std):
        noise = np.random.normal(0, experiment_noise_std)
        if D_S == 0:
            RE_E = RE * 0.01
        else:
            RE_E = RE
        dC = -A_value * (C_T1 - C_out) + N * q + (C_T2 - C_T1) * RE_E
        C_T1 += dC * hl + noise
        return C_T1

    def N_people_meeting_scenario(states, N, duration):
        start_time = np.random.randint(1, observation_time - duration - 120)
        states[start_time:start_time + duration] = N
        return states

    for i in range(num_door_changes):
        statesR1 = N_people_meeting_scenario(door_states, 1, np.random.randint(115, 215))
        statesR2 = N_people_meeting_scenario(door_states, 1, np.random.randint(115, 215))

    statesR1 = np.zeros(observation_time)
    C_realR1 = np.ones(observation_time) * C_out
    statesR2 = np.zeros(observation_time)
    C_realR2 = np.ones(observation_time) * C_out
    num_occ_changes = np.random.randint(7, 12)
    for i in range(num_occ_changes):
        n_value = np.random.randint(5)
        statesR1 = N_people_meeting_scenario(statesR1, n_value, np.random.randint(115, 215))
        statesR2 = N_people_meeting_scenario(statesR2, n_value, np.random.randint(115, 215))
    # states = N_people_meeting_scenario(states, 2, np.random.randint(150,250))
    df = pd.DataFrame(index=range(observation_time))
    df['N1'] = statesR1
    df['N2'] = statesR2

    door_states = np.ones(door_states.shape[0])

    df['D'] = door_states
    for i in range(1, len(C_realR1)):
        # if i==141:
        #     print('DEBUG!')
        C_realR1[i] = C_N_Tplus1(C_realR1[i - 1], C_realR2[i - 1], statesR1[i], door_states[i], room_exchange_rate,
                                 CO2_tres_1)
        C_realR2[i] = C_N_Tplus1(C_realR2[i - 1], C_realR1[i - 1], statesR2[i], door_states[i], room_exchange_rate,
                                 CO2_tres_2)
    df['C_real1'] = C_realR1
    df['C_real2'] = C_realR2

    if len(hl_value_split) > 1:
        if len(room_exchange_rate_split) > 1:
            df.to_csv("two_room_artificial_hl_" + hl_value_split[0] + "_" + hl_value_split[1] + "_exchange_" +
                room_exchange_rate_split[0] + "_" + room_exchange_rate_split[1] + ".csv", index=False)
        else:
            df.to_csv("two_room_artificial_hl_" + hl_value_split[0] + "_" + hl_value_split[1] + "_exchange_" +
                room_exchange_rate_split[0] + ".csv", index=False)
    else:
        if len(room_exchange_rate_split) > 1:
            df.to_csv("two_room_artificial_hl_" + hl_value_split[0] + "_exchange_" +
                room_exchange_rate_split[0] + "_" + room_exchange_rate_split[1] + ".csv", index=False)
        else:
            df.to_csv("two_room_artificial_hl_" + hl_value_split[0] + "_exchange_" +
                room_exchange_rate_split[0] + ".csv", index=False)

    fig = go.Figure()
    fig.update_layout(width=1000, height=500, title=f'CO2 dynamics '+"hl: "+str(hlInput)+" ae: "+str(aeInput),
                      yaxis=dict(title_text="CO2, ppm"), xaxis=dict(title_text="Time, minutes"))

    fig.add_trace(go.Scatter(x=df.index, y=df['C_real1'], name='CO2 R1'))
    fig.add_trace(go.Scatter(x=df.index, y=df['C_real2'], name='CO2 R2'))
    fig.add_trace(go.Bar(x=df.index, y=df['N1'] * 100, name='number of occupants R1', text=statesR1))
    fig.add_trace(go.Bar(x=df.index, y=df['N2'] * 100, name='number of occupants R2', text=statesR2))
    fig.add_trace(go.Bar(x=df.index, y=(df['D'] - 2) * 100, name='Door status', text=door_states))

    fig.show()

aeValues = [0.3,0.1,0.01,0.001,0.0001]

for hlValIndex in range(5):
    for aeIndex in range(5):
        genData(1+0.5*hlValIndex, aeValues[aeIndex])

print("!!!")