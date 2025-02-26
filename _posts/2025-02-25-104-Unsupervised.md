---
title: 8차시 4:머신러닝(비지도학습 및 강화학습) 
layout: single
classes: wide
categories:
  - 머신러닝
tags:
  - 비지도학습
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---


![지도학습](/assets/images/supervised.png)
- 비지도학습은 레이블이 없는 데이터에서 패턴을 찾고, 강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습합니다.



## **1. 비지도학습 (Unsupervised Learning)**

### **1-1. 클러스터링 (Clustering)**

#### **(1) K-Means**
**핵심 개념**:  
데이터를 k개의 클러스터로 그룹화하며, 각 클러스터의 중심점(centroid)과 데이터 포인트 간의 거리를 최소화합니다.

**샘플 예제**:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# 모델 학습
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# 클러스터 할당
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 시각화
plt.scatter([x[0] for x in X], [x[1] for x in X], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.show()
```

---

#### **(2) Hierarchical Clustering**
**핵심 개념**:  
계층적 트리 구조를 통해 데이터를 클러스터링합니다. 병합 기반(Agglomerative) 또는 분할 기반(Divisive) 방법을 사용합니다.

**샘플 예제**:
```python
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# 계층적 클러스터링
model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
labels = model.fit_predict(X)

# 덴드로그램 시각화
linked = linkage(X, 'ward')
dendrogram(linked)
plt.show()
```

---

#### **(3) Gaussian Mixture Model (GMM)**
**핵심 개념**:  
데이터가 여러 개의 가우시안 분포 혼합으로 구성된다고 가정하고, 각 분포의 매개변수를 추정하여 클러스터링합니다.

**샘플 예제**:
```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# GMM 모델 학습
gmm = GaussianMixture(n_components=2)
gmm.fit(X)

# 클러스터 할당
labels = gmm.predict(X)

# 시각화
plt.scatter([x[0] for x in X], [x[1] for x in X], c=labels, cmap='viridis')
plt.show()
```

---

### **1-2. 차원 축소 (Dimensionality Reduction)**

#### **(1) Principal Component Analysis (PCA)**
**핵심 개념**:  
데이터의 분산을 최대한 보존하면서 차원을 줄이는 선형 변환 기법입니다.

**샘플 예제**:
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# PCA 적용
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced))
plt.show()
```

---

#### **(2) Kernel PCA**
**핵심 개념**:  
커널 함수를 사용하여 비선형 데이터를 고차원 공간으로 매핑한 후 PCA를 적용합니다.

**샘플 예제**:
```python
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# Kernel PCA 적용
kpca = KernelPCA(n_components=1, kernel='rbf')
X_reduced = kpca.fit_transform(X)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced))
plt.show()
```

---

#### **(3) Linear Discriminant Analysis (LDA)**
**핵심 개념**:  
클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하여 차원을 줄이는 기법입니다.

**샘플 예제**:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# 데이터 생성
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
y = [0, 0, 1, 1, 0, 1]

# LDA 적용
lda = LDA(n_components=1)
X_reduced = lda.fit_transform(X, y)

# 시각화
plt.scatter(X_reduced, [0] * len(X_reduced), c=y, cmap='viridis')
plt.show()
```

---

## **2. 강화학습 (Reinforcement Learning)**

### **2-1. Upper Confidence Bound (UCB)**
**핵심 개념**:  
탐색(Exploration)과 활용(Exploitation)의 균형을 유지하며 최적의 행동을 선택합니다.

**샘플 예제**:
```python
import numpy as np

# 초기화
num_bandits = 4
num_steps = 100
rewards = np.random.normal(loc=[1, 2, 3, 4], scale=1, size=num_bandits)
Q = np.zeros(num_bandits)
N = np.ones(num_bandits)  # 방문 횟수
c = 2  # 탐색 파라미터

# UCB 알고리즘
for step in range(num_steps):
    ucb_values = Q + c * np.sqrt(np.log(step + 1) / N)
    action = np.argmax(ucb_values)
    reward = np.random.normal(rewards[action])
    Q[action] += (reward - Q[action]) / N[action]
    N[action] += 1

print("최종 선택:", np.argmax(Q))
```

---

### **2-2. Thomson Sampling**
**핵심 개념**:  
베이지안 접근법을 사용하여 각 행동의 확률 분포를 업데이트하며 최적의 행동을 선택합니다.

**샘플 예제**:
```python
import numpy as np

# 초기화
num_bandits = 4
num_steps = 100
rewards = np.random.normal(loc=[1, 2, 3, 4], scale=1, size=num_bandits)
successes = np.ones(num_bandits)
failures = np.ones(num_bandits)

# Thomson Sampling 알고리즘
for step in range(num_steps):
    samples = np.random.beta(successes, failures)
    action = np.argmax(samples)
    reward = np.random.normal(rewards[action])
    if reward > 0:
        successes[action] += 1
    else:
        failures[action] += 1

print("최종 선택:", np.argmax(successes / (successes + failures)))
```

---

### **2-3. Q-Learning**
**핵심 개념**:  
에이전트가 상태와 행동에 대한 Q값을 업데이트하며 최적의 정책을 학습합니다.

**샘플 예제**:
```python
import numpy as np

# 초기화
num_states = 6
num_actions = 3
Q = np.zeros((num_states, num_actions))
gamma = 0.9  # 할인율
alpha = 0.1  # 학습률

# Q-Learning 알고리즘
for episode in range(100):
    state = np.random.randint(num_states)
    while True:
        action = np.argmax(Q[state] + np.random.randn(1, num_actions) * 0.1)
        next_state = np.random.randint(num_states)
        reward = np.random.randn()
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if state == num_states - 1:  # 종료 상태
            break

print("학습된 Q값:\n", Q)
```

---

### **2-4. Deep Q-Learning**
**핵심 개념**:  
신경망을 사용하여 Q값을 근사화하며, 대규모 상태 공간에서 효과적으로 학습합니다.

**샘플 예제**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 신경망 정의
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(2)  # 출력: 각 행동의 Q값
])

# 학습 데이터
state = np.random.rand(1, 4)
action = np.random.randint(2)
reward = np.random.randn()
next_state = np.random.rand(1, 4)

# Q값 업데이트
target = reward + 0.9 * np.max(model(next_state))
with tf.GradientTape() as tape:
    loss = tf.reduce_mean((model(state)[0][action] - target) ** 2)
grads = tape.gradient(loss, model.trainable_variables)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("업데이트된 Q값:", model(state))
```

---

### **2-5. A3C (Asynchronous Advantage Actor-Critic)**
**핵심 개념**:  
여러 에이전트가 비동기적으로 환경과 상호작용하며 정책(Policy)과 가치(Value)를 동시에 학습합니다.

**샘플 예제**:
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

# 신경망 정의
class ActorCritic(Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.policy = Dense(2, activation='softmax')  # 행동 확률
        self.value = Dense(1)  # 상태 가치

    def call(self, inputs):
        x = self.fc1(inputs)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

# 모델 및 옵티마이저 초기화
model = ActorCritic()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 학습 데이터
state = np.random.rand(1, 4)
action = np.random.randint(2)
reward = np.random.randn()

# A3C 알고리즘
with tf.GradientTape() as tape:
    policy, value = model(state)
    advantage = reward - value
    loss = -tf.math.log(policy[0][action]) * advantage + advantage ** 2
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("업데이트된 정책:", policy)
```

---

**결론**:  
비지도학습과 강화학습은 데이터의 특성과 문제 유형에 따라 적합한 모델을 선택해야 합니다. 클러스터링과 차원 축소는 데이터 구조를 이해하거나 전처리에 유용하며, 강화학습은 순차적 의사결정 문제에 적합합니다.