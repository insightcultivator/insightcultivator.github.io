---
title: 9차시 2:딥러닝 기초(실습)
layout: single
classes: wide
categories:
  - 딥러닝
tags:
  - ANN
  - CNN
  - RNN
---


## 1. MNIST 데이터셋을 이용한 숫자 인식
### 1.1 데이터 준비
```python
# TensorFlow 예시
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 데이터 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

- 이 코드는 **TensorFlow**를 사용하여 **MNIST 손글씨 데이터셋**을 불러오고 전처리하는 과정입니다.
    1. **데이터 로드**: `mnist.load_data()`를 사용하여 MNIST 데이터셋을 불러옵니다.  
    2. **정규화 (Normalization)**: 입력 이미지 데이터를 `0~255 → 0~1` 범위로 조정하여 신경망 학습을 원활하게 합니다.  
    3. **원-핫 인코딩 (One-Hot Encoding)**: 레이블을 원-핫 벡터로 변환하여 다중 클래스 분류 문제에 적합한 형식으로 변환합니다.
            - to_categorical(y_train, 10)을 적용하면 각 정수를 길이 10짜리 벡터로 변환.

---
### 1.2 간단한 MLP 모델 구현
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 요약
model.summary()
```

- 이 코드는 **신경망 모델을 구성, 컴파일, 요약 출력**하는 TensorFlow Keras 코드입니다. 

    1. **모델 구성 (Sequential API 사용)**  
        - `Flatten(input_shape=(28, 28))`: 28x28 입력 이미지를 1D 벡터로 변환  
        - `Dense(128, activation='relu')`: 128개 뉴런의 완전 연결층 (ReLU 활성화 함수)  
        - `Dropout(0.2)`: 20% 드롭아웃(과적합 방지)  
        - `Dense(64, activation='relu')`: 64개 뉴런의 완전 연결층  
        - `Dense(10, activation='softmax')`: 10개 클래스에 대한 확률 출력층 (다중 분류)  
    2. **모델 컴파일**  
        - `optimizer='adam'`: Adam 옵티마이저 사용  
        - `loss='categorical_crossentropy'`: 다중 분류를 위한 손실 함수  
        - `metrics=['accuracy']`: 정확도를 평가 지표로 설정  
    3. **모델 요약 (`model.summary()`)**  
        - 전체 네트워크 구조와 파라미터 개수를 출력  

    - 이 모델은 **28x28 크기의 이미지(예: MNIST) 분류를 위한 다층 퍼셉트론(MLP)** 입니다.

---

### 1.3 모델 훈련 및 평가
```python
# 모델 훈련
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'테스트 정확도: {test_acc:.4f}')
```

- 이 코드는 딥러닝 모델을 훈련하고 평가하는 과정의 핵심적인 부분을 담고 있습니다.

    1. **모델 훈련 (`model.fit`)**  
        - `x_train, y_train`을 이용해 모델을 학습  
        - 총 5번(`epochs=5`) 반복 학습  
        - 한 번에 64개(`batch_size=64`) 샘플씩 처리  
        - 훈련 데이터의 20%를 검증(`validation_split=0.2`)에 사용  
    2. **모델 평가 (`model.evaluate`)**  
        - `x_test, y_test`를 사용해 모델 성능 측정  
        - 테스트 데이터에 대한 손실(`test_loss`)과 정확도(`test_acc`) 계산  
        - 최종 테스트 정확도 출력

--- 

### 1.4 결과 시각화
```python
import matplotlib.pyplot as plt

# 학습 곡선 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('모델 정확도')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('모델 손실')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

- 이 코드는 **딥러닝 모델 학습 과정의 성능 변화를 시각화**하는 역할을 합니다. 

    1. **학습 곡선 시각화**  
        - `plt.figure(figsize=(12, 4))`: 그래프 크기 설정  

    2. **정확도(Accuracy) 그래프**  
        - `plt.subplot(1, 2, 1)`: 첫 번째 그래프 영역 설정  
        - `plt.plot(history.history['accuracy'], label='train')`: 학습 데이터 정확도 그래프  
        - `plt.plot(history.history['val_accuracy'], label='validation')`: 검증 데이터 정확도 그래프  
        - `plt.title('모델 정확도')`: 그래프 제목  

    3. **손실(Loss) 그래프**  
        - `plt.subplot(1, 2, 2)`: 두 번째 그래프 영역 설정  
        - `plt.plot(history.history['loss'], label='train')`: 학습 데이터 손실 그래프  
        - `plt.plot(history.history['val_loss'], label='validation')`: 검증 데이터 손실 그래프  
        - `plt.title('모델 손실')`: 그래프 제목  

    4. **마무리 설정**  
        - `plt.legend()`: 범례 추가  
        - `plt.tight_layout()`: 그래프 간격 조정  
        - `plt.show()`: 그래프 출력  


## 2. CNN을 이용한 이미지 분류 실습 
### 2.1 CIFAR-10 데이터셋 소개
```python
from tensorflow.keras.datasets import cifar10

# 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```

- 이 코드는 **CIFAR-10 데이터셋을 로드하고 전처리하는 과정**을 포함한다. 
1. **데이터셋 로드:** `cifar10.load_data()`를 사용하여 10개 클래스로 구성된 이미지 데이터셋(CIFAR-10)을 불러옴.  
2. **정규화:** `x_train`과 `x_test`를 255.0으로 나누어 픽셀 값을 0~1 범위로 스케일링.  
3. **원-핫 인코딩:** `to_categorical()`을 사용해 레이블(`y_train`, `y_test`)을 원-핫 벡터로 변환.  
4. **클래스 이름 정의:** CIFAR-10의 10개 클래스(비행기, 자동차, 새, 고양이 등)를 리스트로 저장.  

---

### 2.2 CNN 모델 구현
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# CNN 모델 구성
cnn_model = Sequential([
    # 첫 번째 합성곱 블록
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # 두 번째 합성곱 블록
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # 분류기
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 모델 컴파일
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 요약
cnn_model.summary()
```

- 이 코드는 **CNN(합성곱 신경망, Convolutional Neural Network)** 모델을 정의하는 Keras 기반 TensorFlow 코드입니다.

    1. **입력 형태**  
        - `(32, 32, 3)` 크기의 컬러 이미지(예: CIFAR-10) 처리.

    2. **합성곱(Conv2D) & 활성화 함수(ReLU)**  
        - 3×3 필터를 사용하여 특징을 추출.  
        - `relu` 활성화 함수로 비선형성을 추가.  
        - `padding='same'`을 적용하여 출력 크기 유지.

    3. **최대 풀링(MaxPooling2D)**  
        - 2×2 풀링으로 공간 크기 절반 축소 → 계산량 감소.

    4. **드롭아웃(Dropout)**  
        - 과적합 방지를 위해 일부 뉴런을 랜덤하게 비활성화.  
        - 합성곱 블록에서는 `0.25`, 완전 연결층에서는 `0.5`.

    5. **완전 연결층(Dense) & 분류기**  
        - `Flatten()`으로 2D 출력을 1D로 변환.  
        - 512개 뉴런의 은닉층(`ReLU` 활성화) 포함.  
        - 마지막 층에서 `softmax` 활성화로 10개 클래스 분류.

    6. **컴파일**  
        - `adam` 옵티마이저 사용 (효율적인 학습).  
        - `categorical_crossentropy` 손실 함수 사용 (다중 클래스 분류).  
        - `accuracy` 평가 지표 사용.

- 즉, 이 모델은 **32×32 컬러 이미지를 입력받아 10개 클래스로 분류하는 CNN 모델**이며, **두 개의 합성곱 블록과 완전 연결층을 포함**하는 구조입니다.

---

### 2.3 모델 훈련 및 평가
```python
# 모델 훈련 (시간 제약으로 epoch 수 조정)
cnn_history = cnn_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# 모델 평가
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test)
print(f'CNN 테스트 정확도: {cnn_test_acc:.4f}')
```

- 이 코드는 CNN(Convolutional Neural Network) 모델을 훈련하고 평가하는 과정의 핵심 부분입니다.  

    1. **모델 훈련 (`fit` 메서드)**  
    - **입력 데이터**: `x_train` (훈련 데이터), `y_train` (정답 레이블)  
    - **훈련 설정**:  
        - `epochs=10` → 10번 반복 학습 (시간 제약 고려)  
        - `batch_size=64` → 64개 샘플씩 미니배치 학습  
        - `validation_split=0.2` → 훈련 데이터의 20%를 검증 데이터로 사용  

    2. **모델 평가 (`evaluate` 메서드)**  
    - 테스트 데이터(`x_test`, `y_test`)로 모델 성능 측정  
    - **출력**: `cnn_test_acc` (테스트 정확도) 및 `cnn_test_loss` (손실 값)  
    - 정확도를 출력하여 모델 성능 확인  


---

### 2.4 예측 및 시각화
```python
# 예측
predictions = cnn_model.predict(x_test)

# 예측 결과 시각화
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i])
    plt.title(f"예측: {class_names[np.argmax(predictions[i])]}\n실제: {class_names[np.argmax(y_test[i])]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

- CNN 모델의 예측 결과를 실제 정답과 함께 시각적으로 확인하는 코드

    1. **CNN 모델 예측 수행**  
    `cnn_model.predict(x_test)`를 통해 테스트 데이터(`x_test`)에 대한 예측 결과(`predictions`)를 얻음.

    2. **예측 결과 시각화**  
    - 3×3 격자로 첫 9개 샘플을 출력.  
    - `plt.imshow(x_test[i])`로 원본 이미지를 표시.  
    - `np.argmax(predictions[i])`를 사용해 가장 높은 확률을 가진 클래스(예측 결과)를 찾음.  
    - 실제 정답(`y_test`)과 비교하여 제목에 표시.  

