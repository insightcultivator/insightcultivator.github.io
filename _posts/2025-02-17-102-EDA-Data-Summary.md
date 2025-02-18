---
title: 5차시 2:탐색적 데이터 분석(EDA)- 데이터 요약 및 시각화
layout: single
classes: wide
categories:
  - 데이터 요약
tags:
  - EDA
---


## **1.데이터 요약**

#### **(1) 수치형 데이터 요약**
- **평균(Mean):** 데이터 값들의 합을 데이터 개수로 나눈 값. 데이터의 중심 경향을 나타냅니다.  
- **중앙값(Median):** 데이터를 정렬했을 때 중간에 위치한 값. 이상치에 덜 민감합니다.  
- **최빈값(Mode):** 데이터에서 가장 자주 등장하는 값.  
- **분산(Variance):** 데이터 값들이 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.  
- **표준편차(Standard Deviation):** 분산의 제곱근. 데이터의 퍼짐 정도를 나타냅니다.  
- **사분위수(Quartiles):** 데이터를 4등분한 값(Q1, Q2=중앙값, Q3). 데이터의 분포를 파악하는 데 유용.

  ```python

    import pandas as pd

    # 데이터 로드
    df = pd.read_csv('titanic.csv')

    # Age 변수 요약
    age = df['age'].dropna()  # 결측값 제거

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

- 왜도(Skewness)와 첨도(Kurtosis)**  
  - **왜도(Skewness):** 데이터 분포의 비대칭성을 나타내는 지표입니다.  
    - 왜도 = 0: 좌우 대칭인 정규분포  
    - 왜도 > 0: 오른쪽 꼬리가 긴 분포 (양의 왜도)  
    - 왜도 < 0: 왼쪽 꼬리가 긴 분포 (음의 왜도)  
  - **첨도(Kurtosis):** 데이터 분포의 뾰족함과 꼬리 두께를 나타내는 지표입니다.  
    - 첨도 = 3: 정규분포 (메소쿠르틱)  
    - 첨도 > 3: 뾰족하고 두꺼운 꼬리를 가진 분포 (렙토크루틱)  
    - 첨도 < 3: 납작하고 얇은 꼬리를 가진 분포 (플라티쿠르틱)  

  ```python
  from scipy.stats import skew, kurtosis

  # 왜도와 첨도 계산
  skewness_age = skew(age)
  kurtosis_age = kurtosis(age, fisher=False)  # 실제 첨도 값 (정규분포 기준 3)

  print(f"왜도: {skewness_age}")
  print(f"첨도: {kurtosis_age}")
  ```

  - **활용:**  
    - 왜도와 첨도를 통해 데이터의 분포 형태를 파악하고, 이상치나 비대칭성을 확인할 수 있습니다.  
    - 예: 왜도가 크게 치우쳐 있다면 데이터 변환이 필요할 수 있습니다. 첨도가 높다면 극단값(outlier)에 주의해야 합니다.



#### **(2) 범주형 데이터 요약**
 범주형 데이터는 특정 카테고리로 분류된 데이터입니다. 이를 요약하기 위해 빈도수와 상대 빈도수를 계산.  
- **빈도수(Frequency):** 각 카테고리가 몇 번 등장하는지 세는 값.  
- **상대 빈도수(Relative Frequency):** 전체 데이터 대비 해당 카테고리의 비율.  



```python
# 성별 빈도수 계산
sex_counts = df['sex'].value_counts()
sex_relative_freq = df['sex'].value_counts(normalize=True)

print("성별 빈도수:")
print(sex_counts)
print("\n성별 상대 빈도수:")
print(sex_relative_freq)
```

## **2.시각화 도구**
#### 1. Matplotlib (plt)

*   **기본 구조**: Figure (전체 도화지)와 Axes (개별 그래프 영역)로 구성됩니다.
*   **역할**: 그래프의 기본 틀을 만들고, Axes 객체를 통해 개별 그래프를 그립니다.

    ```python
    import matplotlib.pyplot as plt

    # Figure 생성 (도화지 준비)
    fig = plt.figure(figsize=(6, 4))

    # Axes 생성 (스케치북 준비, 1개 행 1개 열을 가진 첫번째 subplot)
    ax = fig.add_subplot(111)

    # 그래프 그리기 (스케치북에 그림 그리기)
    ax.plot([1, 2, 3], [4, 5, 6])

    # 그래프 표시
    plt.show()
    ```

* add_subplot() vs plt.subplots()
  - `add_subplot()` 함수 외에 `plt.subplots()` 함수를 사용하여 Figure와 Axes 객체를 동시에 생성할 수도 있습니다. `plt.subplots()` 함수는 여러 개의 subplot을 한 번에 생성하고, 각 subplot에 대한 Axes 객체를 배열 형태로 반환합니다.

    ```python
    fig, axes = plt.subplots(2, 2) # 2행 2열의 subplot 생성

    axes[0, 0].plot([1, 2, 3], [4, 5, 6]) # 첫 번째 subplot에 그래프 그리기
    axes[0, 1].plot([1, 2, 3], [7, 8, 9]) # 두 번째 subplot에 그래프 그리기


    plt.show()
    ```

#### 2. Pandas plot()

*   **역할**: Series나 DataFrame 객체에 내장된 함수로, Matplotlib의 pyplot 함수들을 wrapping하여 더 쉽고 간결하게 그래프를 그릴 수 있습니다.
*   **특징**:
    *   데이터를 자동으로 x, y축에 매핑하고, 필요한 경우 데이터 변환을 수행합니다.
    *   `kind` 매개변수로 그래프 종류를 쉽게 선택할 수 있습니다.
    *   Matplotlib의 기능을 그대로 활용할 수 있습니다.
*   **예시**:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Series 객체 생성
    s = pd.Series([1, 2, 3, 4, 5])

    # Series 객체로 선 그래프 그리기
    s.plot(kind='line', title='Series Line Graph')
    plt.show()

    # DataFrame 객체 생성
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                       'B': [2, 4, 6, 8, 10]})

    # DataFrame 객체로 선 그래프 그리기
    df.plot(kind='line', title='DataFrame Line Graph')
    plt.show()
    ```

#### 3. Seaborn (sns)

*   **역할**: Matplotlib을 wrapping하여 더 쉽고 직관적인 인터페이스를 제공하며, 통계적인 정보를 시각적으로 표현하는 데 특화된 기능을 제공합니다.
*   **특징**:
    *   Matplotlib 기반으로 만들어졌기 때문에 Matplotlib의 Figure와 Axes 객체를 그대로 사용.
    *   함수 기반 접근 방식과 Axes-level 접근 방식을 제공합니다.
*   **예시**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 준비
    data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
    df = pd.DataFrame(data)

    # 함수 기반 접근 방식
    sns.countplot(x='x', data=df)
    plt.show()

    # Axes-level 접근 방식
    fig, ax = plt.subplots()
    sns.countplot(x='x', data=df, ax=ax)
    plt.show()
    ```

#### 4.요약

| 기능 | Matplotlib (plt) | Pandas plot() | Seaborn (sns) |
|---|---|---|---|
| 역할 | 기본 틀 제공 | Matplotlib wrapping | Matplotlib wrapping, 통계적 시각화 |
| 사용 | 복잡한 코드 | 간결한 코드 | 더 쉽고 직관적인 코드 |
| 특징 | Figure, Axes 객체 사용 | 데이터 자동 처리 | 다양한 그래프 스타일 제공 |




## **3.데이터 시각화**
#### **(1) 수치형 데이터 시각화**
데이터의 분포를 시각적으로 확인하면 데이터의 형태와 특징을 더 잘 이해할 수 있습니다. 주요 시각화 도구는 다음과 같습니다:

- **히스토그램(Histogram):** 데이터의 빈도 분포를 막대로 표현합니다. 

  ```
  import matplotlib.pyplot as plt
  import seaborn as sns

  # 히스토그램
  plt.figure(figsize=(8, 4))
  sns.histplot(age, kde=True, bins=20)
  plt.title("Age Distribution")
  plt.xlabel("Age")
  plt.ylabel("Frequency")
  plt.show()
  ```

- **박스 플롯(Box Plot):** 데이터의 사분위수와 이상치를 한눈에 보여줍니다.  

  ```
  # 박스 플롯
  plt.figure(figsize=(8, 4))
  sns.boxplot(x=age)
  plt.title("Age Box Plot")
  plt.xlabel("Age")
  plt.show()
  ```

- **밀도 추정(Kernel Density Estimate, KDE):** 데이터의 확률 밀도를 부드럽게 표현합니다.  

  ```
  # 밀도 추정
  plt.figure(figsize=(8, 4))
  sns.kdeplot(age, shade=True)
  plt.title("Age Density Plot")
  plt.xlabel("Age")
  plt.ylabel("Density")
  plt.show()
  ```

- **산점도(Scatter Plot):** 산점도는 두 변수 간의 관계를 시각적으로 확인하는 데 유용합니다

  ```python
  # 산점도 그리기
  plt.figure(figsize=(8, 6))
  sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.7)
  plt.title("Scatter Plot of Age vs Fare")
  plt.xlabel("Age")
  plt.ylabel("Fare")
  plt.legend(title="Legend", loc="upper right")
  plt.show()
  ```

#### **(2) 범주형 데이터 시각화**
범주형 데이터의 비율을 시각적으로 표현하기 위해 파이 차트와 막대 그래프를 사용합니다.  

- **파이 차트(Pie Chart):** 각 카테고리의 비율을 원형으로 표현합니다. 

  ```
  # 파이 차트
  plt.figure(figsize=(6, 6))
  sex_counts.plot.pie(autopct='%1.1f%%', startangle=90)
  plt.title("Sex Distribution (Pie Chart)")
  plt.ylabel("")
  plt.show()
  ```

- **막대 그래프(Bar Chart):** 각 카테고리의 빈도수를 막대로 표현합니다.  

  ```python
  # 막대 그래프
  plt.figure(figsize=(6, 4))
  sns.countplot(x='sex', data=df)
  plt.title("Sex Distribution (Bar Chart)")
  plt.xlabel("Sex")
  plt.ylabel("Count")
  plt.show()
  ```