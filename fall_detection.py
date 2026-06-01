# =====================================================
# AI 기반 예방형 낙상 감지 시스템
# LSTM + Autoencoder
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)

from sklearn.model_selection import train_test_split

# =====================================================
# 1. 데이터 불러오기
# =====================================================
# MediaPipe Pose를 통해 추출한
# 33개 관절의 x, y 좌표 데이터 사용
# 입력 형태 : (30프레임, 66좌표)

X = np.load('X_lstm.npy')
y = np.load('y_lstm.npy')

print("데이터 shape :", X.shape)
print("라벨 shape :", y.shape)

# =====================================================
# 2. 학습 데이터 / 테스트 데이터 분리
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 3. LSTM 기반 낙상 분류 모델
# =====================================================

lstm_model = Sequential([

    LSTM(
        64,
        input_shape=(30, 66)
    ),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        1,
        activation='sigmoid'
    )
])

# =====================================================
# 4. 모델 컴파일
# =====================================================

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =====================================================
# 5. 모델 학습
# =====================================================

history = lstm_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16
)

# =====================================================
# 6. 모델 평가
# =====================================================

loss, acc = lstm_model.evaluate(
    X_test,
    y_test
)

print("테스트 정확도 :", acc)

# =====================================================
# 7. 정확도 그래프
# =====================================================

plt.figure(figsize=(8,4))

plt.plot(history.history['accuracy'])

plt.title('LSTM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

# =====================================================
# 8. Loss 그래프
# =====================================================

plt.figure(figsize=(8,4))

plt.plot(history.history['loss'])

plt.title('LSTM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

# =====================================================
# 9. 정상 데이터만 추출
# =====================================================
# Autoencoder는 정상 행동만 학습

X_normal = X[y == 0]

# =====================================================
# 10. Autoencoder 모델 생성
# =====================================================

autoencoder = Sequential([

    LSTM(
        64,
        activation='relu',
        input_shape=(30, 66),
        return_sequences=False
    ),

    RepeatVector(30),

    LSTM(
        64,
        activation='relu',
        return_sequences=True
    ),

    TimeDistributed(Dense(66))
])

# =====================================================
# 11. Autoencoder 컴파일
# =====================================================

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# =====================================================
# 12. Autoencoder 학습
# =====================================================

autoencoder.fit(
    X_normal,
    X_normal,
    epochs=10,
    batch_size=16
)

# =====================================================
# 13. Reconstruction Error 계산
# =====================================================

X_pred = autoencoder.predict(X_test)

mse = np.mean(
    np.power(X_test - X_pred, 2),
    axis=(1, 2)
)

# =====================================================
# 14. Threshold 계산
# =====================================================

threshold = np.mean(mse) + 2 * np.std(mse)

print("Threshold :", threshold)

# =====================================================
# 15. 이상행동 감지 그래프
# =====================================================

plt.figure(figsize=(10,4))

plt.plot(mse)

plt.axhline(
    y=threshold,
    color='red',
    linestyle='--',
    label='Threshold'
)

plt.title('Autoencoder Reconstruction Error')
plt.xlabel('Sample')
plt.ylabel('MSE')
plt.legend()

plt.show()

# =====================================================
# 16. 이상행동 여부 판단
# =====================================================

anomaly = mse > threshold

# =====================================================
# 17. LSTM 낙상 예측
# =====================================================

y_pred = lstm_model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

# =====================================================
# 18. 최종 결과 판단
# =====================================================
# 정상
# 위험 행동
# 낙상

final_result = []

for i in range(len(X_test)):

    if anomaly[i] and y_pred[i] == 1:
        final_result.append("낙상")

    elif anomaly[i]:
        final_result.append("위험 행동")

    else:
        final_result.append("정상")

# =====================================================
# 19. 결과 출력
# =====================================================

print("\n===== 최종 판별 결과 =====\n")

for i in range(min(20, len(final_result))):

    print(
        f"{i+1}번째 결과 : {final_result[i]}"
    )