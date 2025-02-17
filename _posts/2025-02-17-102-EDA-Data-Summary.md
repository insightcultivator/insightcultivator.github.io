---
title: 5차시 2:탐색적 데이터 분석(EDA)- 데이터 요약
layout: single
classes: wide
categories:
  - 데이터 요약
tags:
  - EDA
---


## **1. 수치형 데이터 요약**

수치형 데이터는 연속적 또는 이산적인 값을 가지며, 이를 요약하기 위해 다양한 통계량을 사용합니다. 주요 통계량은 다음과 같습니다:  

### **(1) 통계량 계산**

- **평균(Mean):** 데이터 값들의 합을 데이터 개수로 나눈 값. 데이터의 중심 경향을 나타냅니다.  
- **중앙값(Median):** 데이터를 정렬했을 때 중간에 위치한 값. 이상치에 덜 민감합니다.  
- **최빈값(Mode):** 데이터에서 가장 자주 등장하는 값.  
- **분산(Variance):** 데이터 값들이 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.  
- **표준편차(Standard Deviation):** 분산의 제곱근. 데이터의 퍼짐 정도를 나타냅니다.  
- **사분위수(Quartiles):** 데이터를 4등분한 값(Q1, Q2=중앙값, Q3). 데이터의 분포를 파악하는 데 유용.

<br>

```python

  import pandas as pd

  # 데이터 로드
  df = pd.read_csv('titanic.csv')

  # Age 변수 요약
  age = df['Age'].dropna()  # 결측값 제거

  # 통계량 계산
  mean_age = age.mean()
  median_age = age.median()
  mode_age = age.mode()[0]
  variance_age = age.var()
  std_age = age.std()
  quartiles_age = age.quantile([0.25, 0.5, 0.75])
  
  print(f"평균 나이: {mean_age}")
  print(f"중앙값 나이: {median_age}")
  print(f"최빈값 나이: {mode_age}")
  print(f"분산: {variance_age}")
  print(f"표준편차: {std_age}")
  print(f"사분위수: {quartiles_age}")
    
```
<br>

### **(2) 분포 시각화**
데이터의 분포를 시각적으로 확인하면 데이터의 형태와 특징을 더 잘 이해할 수 있습니다. 주요 시각화 도구는 다음과 같습니다:  

- **히스토그램(Histogram):** 데이터의 빈도 분포를 막대로 표현합니다.  
- **박스 플롯(Box Plot):** 데이터의 사분위수와 이상치를 한눈에 보여줍니다.  
- **밀도 추정(Kernel Density Estimate, KDE):** 데이터의 확률 밀도를 부드럽게 표현합니다.  

<br>

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 히스토그램
plt.figure(figsize=(8, 4))
sns.histplot(age, kde=True, bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 박스 플롯
plt.figure(figsize=(8, 4))
sns.boxplot(x=age)
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.show()

# 밀도 추정
plt.figure(figsize=(8, 4))
sns.kdeplot(age, shade=True)
plt.title("Age Density Plot")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()
```

<br>

## **2. 범주형 데이터 요약**

### **(1) 빈도수 및 비율 계산**
 범주형 데이터는 특정 카테고리로 분류된 데이터입니다. 이를 요약하기 위해 빈도수와 상대 빈도수를 계산.  
- **빈도수(Frequency):** 각 카테고리가 몇 번 등장하는지 세는 값.  
- **상대 빈도수(Relative Frequency):** 전체 데이터 대비 해당 카테고리의 비율.  

<br>

```python
# 성별 빈도수 계산
sex_counts = df['Sex'].value_counts()
sex_relative_freq = df['Sex'].value_counts(normalize=True)

print("성별 빈도수:")
print(sex_counts)
print("\n성별 상대 빈도수:")
print(sex_relative_freq)
```

<br>


### **(2) 시각화(파이 차트, 막대 그래프)**
범주형 데이터의 비율을 시각적으로 표현하기 위해 파이 차트와 막대 그래프를 사용합니다.  

- **파이 차트(Pie Chart):** 각 카테고리의 비율을 원형으로 표현합니다.  
- **막대 그래프(Bar Chart):** 각 카테고리의 빈도수를 막대로 표현합니다.  

<br>

```python
# 파이 차트
plt.figure(figsize=(6, 6))
sex_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Sex Distribution (Pie Chart)")
plt.ylabel("")
plt.show()

# 막대 그래프
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df)
plt.title("Sex Distribution (Bar Chart)")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()
```

<br>



## **3. GenAI 활용:**  
   - AI 도구를 활용해 데이터 요약 리포트를 자동으로 생성하는 시연을 보여줍니다.  
   - 예: "이 데이터셋의 주요 통계량과 시각화를 요약해줘"라고 요청하면, AI가 기본적인 요약 결과를 제공합니다.  
