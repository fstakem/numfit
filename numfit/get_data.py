from fitparse import FitFile
import pandas as pd
import numpy as np

from scipy.fftpack import fft, fftfreq, fftshift

import matplotlib.pyplot as plt

data_path = './data/raw/12_29_17_leg_day.fit'
t_ms = 50
data = FitFile(data_path)
msgs = data.get_messages()

var_names = ['timestamp', 'heart_rate']
parsed_data = { n: [] for n in var_names}


for msg in msgs:
    fields = [f for f in msg.fields if f.name in var_names]
    field_names = [f.name for f in fields]

    if field_names == var_names:
        select_fields = [(f.name, f.value) for f in fields if f.value != None]
        
        if len(select_fields) == len(var_names):
            for name, value in select_fields:
                parsed_data[name].append(value)

df = pd.DataFrame()
df['hr'] = np.array(parsed_data['heart_rate'])
df.index = np.array(parsed_data['timestamp'])
df.hr = pd.to_numeric(df.hr)
df_resample = df.resample('{}L'.format(t_ms))
df = df_resample.interpolate()

num_samples = len(df)
fs = 1 / t_ms * 1000
y = abs(fft(df.hr)) * (1 / num_samples)
x = fftfreq(num_samples, 1 / fs)
x = fftshift(x)
y = fftshift(y)

df_freq = pd.DataFrame()

import ipdb
ipdb.set_trace()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df.index, df.hr)
ax.set_title('Title')
ax.set_xlabel('time')
ax.set_ylabel('hr')

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(x, y)
ax1.set_xlim(-.0005, .005)

plt.show()
