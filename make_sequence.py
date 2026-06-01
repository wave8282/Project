import pandas as pd
import numpy as np

# 👉 CSV 불러오기
data = pd.read_csv('D:/pose_project/fall_data.csv', header=None)

sequence_length = 30
sequences = []

# 👉 30프레임씩 묶기
for i in range(len(data) - sequence_length):
    seq = data.iloc[i:i+sequence_length].values
    sequences.append(seq)

# 👉 numpy 배열 변환
X = np.array(sequences)

# 👉 라벨 생성 (예: 걷기 = 0)
y = np.zeros(len(X))

# 👉 결과 출력
print("X shape:", X.shape)
print("y shape:", y.shape)

# 👉 파일 저장 (중요🔥)
np.save('D:/pose_project/fall_X.npy', X)
np.save('D:/pose_project/y.npy', y)

print("저장 완료!")