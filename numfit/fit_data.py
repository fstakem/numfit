import pandas as pd
import numpy as np

from fitparse import FitFile


class FitData(object):

    def get_df(self, path, T_ms, var_mapping):
        var_names = [x[0] for x in var_mapping]
        parsed_data = self.parse_data(path, var_names)
        df = pd.DataFrame()

        for input, output in var_mapping:
            if output != 'index':
                df[output] = np.array(parsed_data[input])
                df[output] = pd.to_numeric(df[output])
            else:
                df.index = np.array(parsed_data[input])

        df = self.resample_df(df, T_ms)

        return df

    def parse_data(self, path, var_names):
        raw_data = FitFile(path)
        msgs = raw_data.get_messages()
        parsed_data = { n: [] for n in var_names}

        for msg in msgs:
            fields = [f for f in msg.fields if f.name in var_names]
            field_names = [f.name for f in fields]

            if set(field_names) == set(var_names):
                select_fields = [(f.name, f.value) for f in fields if f.value != None]
                
                if len(select_fields) == len(var_names):
                    for name, value in select_fields:
                        parsed_data[name].append(value)

        return parsed_data

    def resample_df(self, df, T_ms):
        df_resample = df.resample('{}L'.format(T_ms))
        df = df_resample.interpolate()

        return df


if __name__ == '__main__':
    path = './data/raw/12_15_17_cardio_day.fit'
    T_ms = 50
    var_mapping = ( ('heart_rate', 'hr'),
                    ('timestamp', 'index'))

    fit_data = FitData()
    df = fit_data.get_df(path, T_ms, var_mapping)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.index, df.hr)
    ax.set_title('Title')
    ax.set_xlabel('time')
    ax.set_ylabel('hr')

    plt.show()