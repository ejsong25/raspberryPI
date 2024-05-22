import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def dropDirtyData(df):
    # -inf와 inf 값을 NaN으로 대체
    df.replace([float('-inf'), float('inf')], float('nan'), inplace=True)

    # NaN이 포함된 행을 삭제
    df = df.dropna(axis=0, how='any')
    print(df)


# 각 서브캐리어 값을 시간에 따라 플롯
motion = ['sit', 'stand', 'walk', 'fall', 'empty']
for i in motion:
    df = pd.read_csv(f"csvData/output_{i}.csv")
    for column in range(5):
        # 시간 시리즈 생성 (데이터의 행 수만큼)
        time_series = range(len(df))
        # 그래프 크기 설정
        # plt.figure(figsize=(14, 8))
        plt.plot(time_series, df.iloc[:, column])
        # 그래프 라벨 설정
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(i)

        # 범례 위치 설정
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 그리드 및 레이아웃 설정
        plt.grid(True)
        plt.tight_layout()

    # 그래프 표시
    plt.show()

# for i in range(0, 64):
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(
    #     df[f'Sub {i} RXTX 0/0'].values.reshape(1, -1))
    # plt.plot(df.index, df_scaled)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title(f'stand')
    # plt.show()

"""
b = ['sit','empty','fall','walk','stand']
for bs in b:
    print(bs)
    a(bs)
"""
