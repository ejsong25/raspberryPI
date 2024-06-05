import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression

def moving_avarage_smoothing(X, k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S

# 복소수 문자열을 진폭값으로 변환한다


def calculate_amplitude(complex_str):
    return abs(complex(complex_str))

# DataFrame 의 모든 열에 대하여 복소수 문자열을 진폭값으로 변환한다


def rolling_k(df, k):
    for col in df.columns:
        df[col] = df[col].rolling(window=k).mean()
    return df


def apply_calculate_amplitude(df):
    for col in df.columns:
        df[col] = df[col].apply(calculate_amplitude)
    return df

# CSI 시계열 데이터 중 모든 부반송파 데이터용 서브캐리어에 대해 시각화 한다


def print_output(motion, df):
    for col in df.columns:
        plt.plot(df.index.values, df[col])
    plt.title(motion)
    plt.show()


# 제거해야 할 열 인덱스 : 맥주소, 시간, 부반송파 미할당 서브캐리어 번호
to_drop = ['mac', 'time', '_0', '_1', '_2', '_3', '_11', '_25',
           '_32', '_39', '_43', '_60', '_61', '_62', '_63']

motion = ['empty', 'fall', 'sit', 'stand', 'walk']
dic = {}

scaler = StandardScaler()
model = LinearRegression()

for m in motion:
    df = pd.read_csv(
        f"csvData/output_{m}.csv")
    df.drop(columns=to_drop, inplace=True)
    df_ = apply_calculate_amplitude(df)
    df__ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns)
    df___ = pd.DataFrame(gaussian_filter(df__, 100), columns=df__.columns)
    dic[m] = df___

for motion, df in dic.items():
    for col in df.columns:
        index_df = pd.DataFrame(df.index.values)
        model.fit(index_df, df[col])
        
    # plt.plot(df.index.values, df.iloc[:, 0])
    # plt.title(f"{motion}")
    # plt.show()
