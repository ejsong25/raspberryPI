import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

# CSV 파일에서 데이터 읽기
file_path = 'csvData/output_empty.csv'
df = pd.read_csv(file_path)

# 데이터 확인
print(df.head())

# 여기서는 'timestamp'와 'amplitude'라는 컬럼이 있다고 가정합니다.
# 실제 데이터의 컬럼 이름에 맞게 수정해야 합니다.
time_series = range(500)
original_data = df['amplitude']

# 이동 평균 필터 함수
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 이동 평균을 적용한 데이터
window_size = 10  # 윈도우 크기를 적절히 조절
smoothed_data = moving_average(original_data, window_size)

# 가우시안 필터를 적용한 데이터
sigma = 2  # sigma 값을 적절히 조절
smoothed_data_gaussian = gaussian_filter1d(original_data, sigma)

# 그래프 크기 설정
plt.figure(figsize=(12, 8))

# 이동 평균을 적용한 데이터 시각화
plt.plot(time_series[:len(smoothed_data)], smoothed_data, label='Smoothed (Moving Average)', linewidth=2)

# 가우시안 필터를 적용한 데이터 시각화
plt.plot(time_series, smoothed_data_gaussian, label='Smoothed (Gaussian Filter)', linewidth=2)

# 원본 데이터 시각화
plt.plot(time_series, original_data, label='Original', alpha=0.5)

# 그래프 라벨 설정
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('CSI Data Smoothing')

# 범례 위치 설정
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 그리드 및 레이아웃 설정
plt.grid(True)
plt.tight_layout()

# 그래프 표시
plt.show()
