import numpy as np

# 👉 기존에 만든 파일 불러오기
X_normal = np.load('D:/pose_project/X.npy')        # 정상
X_fall   = np.load('D:/pose_project/fall_X.npy')   # 낙상

print("정상:", X_normal.shape)
print("낙상:", X_fall.shape)

# 👉 라벨 생성
y_normal = np.zeros(len(X_normal))   # 0
y_fall   = np.ones(len(X_fall))      # 1

# 👉 데이터 합치기
X_all = np.concatenate([X_normal, X_fall], axis=0)
y_all = np.concatenate([y_normal, y_fall], axis=0)

# 👉 섞기 (중요🔥)
idx = np.random.permutation(len(X_all))
X_all = X_all[idx]
y_all = y_all[idx]

print("최종 X:", X_all.shape)
print("최종 y:", y_all.shape)

# 👉 저장
np.save('D:/pose_project/X_lstm.npy', X_all)
np.save('D:/pose_project/y_lstm.npy', y_all)

print("LSTM 데이터셋 저장 완료!")