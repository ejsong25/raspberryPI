import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(f"csvData/output_stand_1.csv")
for i in range(0, 64):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(
        df[f'Sub {i} RXTX 0/0'].values.reshape(1, -1))
    plt.plot(df.index, df_scaled)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'{a}')
    plt.show()

"""
b = ['sit','empty','fall','walk','stand']
for bs in b:
    print(bs)
    a(bs)
"""
