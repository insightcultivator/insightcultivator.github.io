---
title: 5차시 3:탐색적 데이터 분석(EDA)- 데이터 탐색
layout: single
classes: wide
categories:
  - 데이터 요약
tags:
  - EDA
---


## **3. 데이터 탐색**

### **1. 이상치 분석**
- 데이터에서 일반적인 패턴과 크게 벗어난 값으로, 데이터의 정확성을 해칠 수 있거나 중요한 정보를 제공할 수도 있습니다.
- 이상치는 다음과 같은 이유로 발생할 수 있습니다:
  - 측정 오류
  - 입력 오류
  - 자연스러운 변동성
- 이상치를 탐지하는 주요 방법:
  - **박스 플롯(Box Plot)**: 사분위수와 IQR(Interquartile Range)을 사용하여 이상치를 시각적으로 확인.
  - **Z-Score**: 평균에서 표준편차 몇 배 이상 떨어진 데이터를 이상치로 간주 (z의 기준점 3)
    - 정규분포의 경험적 규칙에 따라, z의 절대값 > 3 큰 데이터는 전체 데이터의 0.3% 미만으로 매우 드물다.
    - 이상치를 식별하기 위한 실용적이고 균형 잡힌 기준이다.
    - 너무 낮거나 높은 기준 대비 적절한 타협점을 제공한다.

  - **IQR 기반 필터링**: Q1 - 1.5 * IQR ~ Q3 + 1.5 * IQR 범위를 벗어나는 값을 이상치로 정의.

  - 실습

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np

  # Titanic 데이터셋 로드
  titanic = sns.load_dataset('titanic')

  # 박스 플롯으로 이상치 확인
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=titanic['age'])
  plt.title("Box Plot of Age")
  plt.show()

  # Z-Score 계산으로 이상치 탐지
  from scipy.stats import zscore

  titanic['age_zscore'] = zscore(titanic['age'].dropna())
  outliers = titanic[(titanic['age_zscore'] > 3) | (titanic['age_zscore'] < -3)]
  print("이상치 데이터:")
  print(outliers[['age', 'age_zscore']])

  # IQR 기반 이상치 탐지
  Q1 = titanic['age'].quantile(0.25)
  Q3 = titanic['age'].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  outliers_iqr = titanic[(titanic['age'] < lower_bound) | (titanic['age'] > upper_bound)]
  print("IQR 기반 이상치 데이터:")
  print(outliers_iqr[['age']])
  ```


### **2. 결측값 분석**
데이터가 누락된 상태를 의미합니다.
- 결측값의 원인:
  - 데이터 수집 과정에서의 문제
  - 설문 응답 미제출
  - 시스템 오류
- 결측값 처리 방법:
  - **삭제**: 결측값이 있는 행 또는 열 제거.
  - **대체**: 평균, 중앙값, 최빈값 등으로 결측값 대체.
  - **보간법**: 시간 순서 데이터에서 이전/다음 값으로 보간.

- **실습**

  ```python
  # 결측값 비율 확인
  missing_values = titanic.isnull().sum() / len(titanic) * 100
  print("결측값 비율(%):")
  print(missing_values)

  # 결측값 시각화
  plt.figure(figsize=(8, 6))
  sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
  plt.title("Missing Values Heatmap")
  plt.show()

  # 결측값 처리: 나이(Age) 평균으로 대체
  titanic['age'].fillna(titanic['age'].mean(), inplace=True)

  # 결측값 처리 후 확인
  print("결측값 처리 후:")
  print(titanic.isnull().sum())
  ```


### **3. 변수 간 관계 분석**
- 변수 간 관계를 분석하면 데이터 내 숨겨진 패턴이나 상관관계를 발견할 수 있습니다.
- 주요 분석 방법:
  - **산점도(Scatter Plot)**: 두 변수 간 선형 또는 비선형 관계 확인.
  - **상관관계 행렬(Correlation Matrix)**: 수치형 변수 간 상관관계 계산.
  - **카테고리별 비교**: 범주형 변수와 수치형 변수 간 관계 확인.

- **실습**

  ```python
  # 산점도로 관계 확인
  plt.figure(figsize=(8, 6))
  sns.scatterplot(data=titanic, x='age', y='fare', hue='survived')
  plt.title("Age vs Fare with Survival")
  plt.show()

  # 상관관계 행렬
  numeric_columns = ['age', 'fare', 'pclass', 'survived']
  correlation_matrix = titanic[numeric_columns].corr() #corr(): 피어슨 상관계수(-1 ~ +1)

  plt.figure(figsize=(8, 6))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
  plt.title("Correlation Matrix")
  plt.show()

  # 카테고리별 비교: 성별(Sex)에 따른 생존률
  sns.barplot(data=titanic, x='sex', y='survived')
  plt.title("Survival Rate by Sex")
  plt.show()
  ```

---

### **4. 패턴/트렌드 발견**
- 데이터에서 반복되는 패턴이나 시간에 따른 트렌드를 발견하면 미래를 예측하거나 인사이트를 도출하는 데 유용합니다.
- 주요 분석 방법:
  - **시계열 분석(Time Series Analysis)**: 시간에 따른 변화 추세 확인.
  - **그룹별 집계(Aggregation by Group)**: 특정 그룹(예: 성별, 클래스) 내에서의 패턴 확인.
  - **클러스터링(Clustering)**: 데이터 포인트를 유사한 그룹으로 묶어 패턴 발견.

- **실습**

  ```python
  # 클래스(Class)별 생존률 비교
  sns.countplot(data=titanic, x='pclass', hue='survived')
  plt.title("Survival Count by Class")
  plt.show()

  # 나이(Age) 그룹별 생존률
  titanic['age_group'] = pd.cut(titanic['age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young', 'Adult', 'Senior'])
  sns.barplot(data=titanic, x='age_group', y='survived') #신뢰구간
  plt.title("Survival Rate by Age Group")
  plt.show()

  # 가족 크기(Family Size) 생성 및 생존률 분석
  titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
  sns.lineplot(data=titanic, x='family_size', y='survived') #신뢰구간
  plt.title("Survival Rate by Family Size")
  plt.show()
  ```

