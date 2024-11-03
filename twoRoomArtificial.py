import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(3)

total_obs_time = 24 #hours

possible_states = np.array([0,1,2,3])

CO2_tres = 1 #time resolution of CO2 sensor, min

#Calculating CO2 emission rate, q

C_co2 = 0.04 # volume concentraiton of CO2 in exhaled air
Vt = 0.5 #L/breath, tidal voulume
Rt = 15 #breaths/min, breathing rate

V_room = 60000 #L, 3x4x5 m3 room volume

q = Rt*Vt*C_co2/V_room*1e6 #ppm/min, CO2 emission rate
print(f"q = {q} ppm/min")

q = int(q) #ppm/min
A = 0.015 #1/min, insulation characteristics of the room
#(I've set A manually so that the equilibrium CO2 concentrations look natural; i.e. 730 ppm for 1 occupant, 1050 ppm for 2 occupants etc.)
C_out = 400 #ppm

room_exchange_rate = 0.1 # air exchange between rooms

signal_noise_std = 10 #ppm
experiment_noise_std = 1 #ppm (See C_N_with_noise function)

observation_time = 3440 #minutes

door_states = np.zeros(observation_time)
num_door_changes = np.random.randint(7,12)

def C_N_Tplus1(C_T1, C_T2, N, D_S, RE, experiment_noise_std = experiment_noise_std):
    noise = np.random.normal(0, experiment_noise_std)
    if D_S==0:
        RE_E = RE*0.01
    else:
        RE_E = RE
    dC = -A*(C_T1-C_out)+N*q+(C_T2-C_T1)*RE_E
    C_T1 += dC*CO2_tres+noise
    return C_T1

def N_people_meeting_scenario(states, N, duration):
    start_time = np.random.randint(1,observation_time-duration-120)
    states[start_time:start_time+duration] = N
    return states

for i in range(num_door_changes):
    statesR1 = N_people_meeting_scenario(door_states, 1, np.random.randint(150, 250))
    statesR2 = N_people_meeting_scenario(door_states, 1, np.random.randint(150, 250))

statesR1 = np.zeros(observation_time)
C_realR1 = np.ones(observation_time)*C_out
statesR2 = np.zeros(observation_time)
C_realR2 = np.ones(observation_time)*C_out
num_occ_changes = np.random.randint(7,12)
for i in range(num_occ_changes):
    n_value = np.random.randint(5)
    statesR1 = N_people_meeting_scenario(statesR1, n_value, np.random.randint(150,250))
    statesR2 = N_people_meeting_scenario(statesR2, n_value, np.random.randint(150, 250))
# states = N_people_meeting_scenario(states, 2, np.random.randint(150,250))
df = pd.DataFrame(index = range(observation_time))
df['N1'] = statesR1
df['N2'] = statesR2
df['D'] = door_states
for i in range(1,len(C_realR1)):
    C_realR1[i] = C_N_Tplus1(C_realR1[i - 1], C_realR2[i - 1], statesR1[i], door_states[i], room_exchange_rate)
    C_realR2[i] = C_N_Tplus1(C_realR2[i - 1], C_realR1[i - 1], statesR2[i], door_states[i], room_exchange_rate)
df['C_real1'] = C_realR1
df['C_real2'] = C_realR2

df.to_csv('two_room_artificial.csv',index=False)

fig = go.Figure()
fig.update_layout(width=1000, height=500, title = f'CO2 dynamics',
                  yaxis=dict(title_text="CO2, ppm"), xaxis=dict(title_text="Time, minutes") )

fig.add_trace(go.Scatter(x=df.index, y=df['C_real1'], name='CO2 R1'))
fig.add_trace(go.Scatter(x=df.index, y=df['C_real2'], name='CO2 R2'))
fig.add_trace(go.Bar(x=df.index, y=df['N1']*100, name='number of occupants R1', text=statesR1))
fig.add_trace(go.Bar(x=df.index, y=df['N2']*100, name='number of occupants R2', text=statesR2))
fig.add_trace(go.Bar(x=df.index, y=(df['D']-2)*100, name='Door status', text=door_states))

fig.show()

