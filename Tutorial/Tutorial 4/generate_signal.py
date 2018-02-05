import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

freq1 = 500
freq2 = 800
phase_diff = np.pi/3

P1 = 0.04
P2 = 0.02

t = np.linspace(0,0.5,10001)

std = 0.01
WN1 = np.random.normal(0, std, size=len(t))
WN2 = np.random.normal(0, std, size=len(t))

signal1 = P1*np.cos(2*np.pi*freq1*t)+WN1
signal2 = P2*np.cos(2*np.pi*freq2*t+phase_diff)+WN2

df1 = pd.DataFrame()
df1['t (s)'] = t
df1['P (Pa)'] = signal1

df2 = pd.DataFrame()
df2['t (s)'] = t
df2['P (Pa)'] = signal2

df1.to_csv('signal1.csv', sep=',', index=False)
df2.to_csv('signal2.csv', sep=',', index=False)

plt.plot(t, signal1)
plt.plot(t, signal2)

