---
title: 8차시 1:머신 러닝(Machin Learning 기초) 
layout: single
classes: wide
categories:
  - 데이터 시각화
tags:
  - maching learning
---

## **1. 머신러닝 개요**

### **1.1 머신러닝이란?**
- **머신러닝의 정의와 중요성**:
  - 데이터에서 패턴을 학습하고 이를 활용해 예측하거나 결정하는 기술.
  - 전통적인 프로그래밍은 명확한 규칙을 기반으로 동작하지만, 머신러닝은 데이터를 통해 규칙을 자동으로 학습.
- **전통적인 프로그래밍과 머신러닝의 차이점**:
  - 전통적 프로그래밍: 입력 + 규칙 → 출력
  - 머신러닝: 입력 + 출력 → 규칙
- **머신러닝의 역사적 발전과 현재 동향**:
  - 1950년대: Alan Turing의 "컴퓨터가 생각할 수 있을까?" 질문.
  - 2000년대 이후: 빅데이터와 딥러닝의 발전.
- **실생활 응용 사례 소개**:
  - 의료 진단(암 분류), 추천 시스템(넷플릭스), 자연어 처리(번역), 자율주행 등.

---

### **1.2 머신러닝의 유형**
- **지도학습 (Supervised Learning)**:
  - **분류(Classification)**: 스팸 메일 분류, 이미지 분류.
  - **회귀(Regression)**: 집값 예측, 주식 가격 예측.
- **비지도학습 (Unsupervised Learning)**:
  - **군집화(Clustering)**: 고객 세그먼테이션, 문서 그룹화.
  - **차원 축소(Dimensionality Reduction)**: 데이터 시각화, 노이즈 제거.
- **강화학습 (Reinforcement Learning)**:
  - 게임 AI, 로봇 제어.
- **준지도학습 (Semi-supervised Learning)**:
  - 일부 레이블이 있는 데이터를 활용.


### **1.3 머신러닝 워크플로우**
1. **문제 정의**: 해결하려는 문제를 명확히 정의.
2. **데이터 수집 및 준비**: 데이터 수집, 결측치 처리, 이상치 제거.
3. **특성 추출 및 선택**: 데이터에서 중요한 특성 추출.
4. **모델 선택 및 학습**: 알고리즘 선택 및 하이퍼파라미터 설정.
5. **모델 평가 및 최적화**: 성능 지표를 사용해 모델 평가.
6. **모델 배포 및 모니터링**: 실제 환경에서 모델 적용.

## **2.주요 머신러닝 알고리즘**

### **2.1 지도학습 알고리즘**
- **선형 회귀 (Linear Regression)**:
  - 기본 원리: 입력 변수와 출력 변수 간 선형 관계를 모델링.
  - 손실 함수: MSE(Mean Squared Error).
  - 최적화: 경사하강법(Gradient Descent).
- **로지스틱 회귀 (Logistic Regression)**:
  - 이진 분류와 다중 분류 가능.
  - 활성화 함수: 시그모이드(Sigmoid).
- **결정 트리 (Decision Trees)**:
  - 정보 이득, 엔트로피, 지니 계수를 사용해 분할.
- **앙상블 기법**:
  - **랜덤 포레스트**: 여러 결정 트리를 조합.
  - **그래디언트 부스팅**: 순차적으로 모델을 강화.



### **2.2 비지도학습 알고리즘**
- **K-평균 군집화 (K-means Clustering)**:
  - 알고리즘 원리: 데이터를 K개의 클러스터로 나눔.
  - 최적의 K 선택: Elbow Method.
- **주성분 분석 (PCA)**:
  - 차원 축소의 필요성: 데이터 시각화, 연산 효율성.
  - PCA의 수학적 직관: 공분산 행렬의 고유값 분해.


### **2.3 모델 평가 방법**
- **분류 평가 지표**:
  - 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수.
  - 혼동 행렬(Confusion Matrix).
  - ROC 곡선과 AUC.
- **회귀 평가 지표**:
  - MAE(Mean Absolute Error), MSE(Mean Squared Error), RMSE, R-squared.
- **교차 검증 (Cross-validation)**:
  - K-Fold Cross Validation.

## **3.머신러닝 실무 고려사항**

### **3.1 특성 공학 (Feature Engineering)**
- **특성 선택의 중요성**: 관련 없는 특성 제거.
- **범주형 변수 처리**: One-Hot Encoding, Label Encoding.
- **스케일링과 정규화**: Min-Max Scaling, Standardization.
- **결측치와 이상치 처리**: 평균 대체, 이상치 제거.

### **3.2 과적합과 과소적합 (10분)**
- **과적합/과소적합의 이해**:
  - 과적합: 학습 데이터에 너무 맞춰짐.
  - 과소적합: 데이터 패턴을 충분히 학습하지 못함.
- **편향-분산 트레이드오프**:
  - 편향(Bias): 모델의 단순성.
  - 분산(Variance): 모델의 복잡성.
- **정규화 기법**: L1(Lasso), L2(Ridge).
- **조기 종료(Early Stopping)**: 학습 중단.


### **3.3 하이퍼파라미터 튜닝**
- **그리드 서치(Grid Search)**: 모든 가능한 조합 탐색.
- **랜덤 서치(Random Search)**: 무작위로 조합 탐색.
- **베이지안 최적화(Bayesian Optimization)**: 확률적 접근.


## **4.실습 - Python을 활용한 머신러닝**

### **4.1 환경 설정 및 라이브러리 소개**

```python
# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
```

### **4.2 지도학습 실습**

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 학습
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# 랜덤 포레스트 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```


### **4.3 비지도학습 실습**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# PCA를 통한 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering with PCA")
plt.show()
```
