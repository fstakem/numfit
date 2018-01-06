from fitparse import FitFile
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data_path = './data/raw/12_29_17_leg_day.fit'
data = FitFile(data_path)
msgs = data.get_messages()

var_names = ['timestamp', 'heart_rate']
parsed_data = { n: [] for n in var_names}


for msg in msgs:
    fields = [f for f in msg.fields if f.name in var_names]
    field_names = [f.name for f in fields]

    if field_names == var_names:
        for f in fields:
            parsed_data[f.name].append(f.value)

df = pd.DataFrame()
df['hr'] = np.array(parsed_data['heart_rate'])
df.index = np.array(parsed_data['timestamp'])

df_freq = pd.DataFrame()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df.index, df.hr)
ax.set_title('Title')
ax.set_xlabel('time')
ax.set_ylabel('hr')
plt.show()
