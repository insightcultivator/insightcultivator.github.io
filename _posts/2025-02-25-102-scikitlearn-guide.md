---
title: 8차시 2:머신 러닝(Scikit-Learn Guide) 
layout: single
classes: wide
categories:
  - 머신러닝
tags:
  - scikit-learn
---

![scikit-learn logo](/assets/images/Scikit_learn_logo_small.svg)
[Scikit-learn의 Getting_Started 페이지](https://scikit-learn.org/stable/getting_started.html)

## **1. Estimator Basics (모델 학습 및 예측)**
- **핵심 개념**: 
  - Scikit-learn의 모든 모델은 `Estimator` 객체로 구현됩니다.
  - `fit()` 메서드를 사용해 데이터에 모델을 학습시키고, `predict()` 메서드로 새로운 데이터를 예측합니다.
- **예제 코드**:
  ```python
  from sklearn.ensemble import RandomForestClassifier

  clf = RandomForestClassifier(random_state=0)
  X = [[1, 2, 3], [11, 12, 13]]  # 샘플 데이터
  y = [0, 1]  # 타겟 레이블
  clf.fit(X, y)  # 모델 학습
  clf.predict([[4, 5, 6], [14, 15, 16]])  # 새로운 데이터 예측
  ```
- **설명 포인트**:
  - `X`: 입력 데이터(특성 행렬), `(n_samples, n_features)` 형태.
  - `y`: 출력 데이터(타겟 값), 분류 문제에서는 클래스 레이블, 회귀 문제에서는 실수 값.
  - 학습된 모델은 새로운 데이터를 예측할 때 다시 학습할 필요가 없습니다.


## **2. Transformers and Pre-processors (데이터 전처리)**
### **2.1. 핵심 개념**:
  - 데이터 전처리는 머신러닝 파이프라인에서 중요한 단계입니다.
  - Scikit-learn은 데이터 변환을 위한 `Transformer` 객체를 제공하며, `fit()`과 `transform()` 메서드를 사용합니다.
  - **예제 코드**:
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = [[0, 15], [1, -10]]
    scaled_X = scaler.fit_transform(X)  # 데이터 표준화
    print(scaled_X)
    ```
  - **설명 포인트**:
    - `StandardScaler`: 데이터를 평균 0, 분산 1로 표준화합니다.
    - 전처리는 데이터의 스케일을 맞추거나 결측치를 처리하는 데 사용됩니다.

### **2.2. fit(), transform(), fit_transform()**:
- fit(): 입력 데이터(X)를 분석하여 필요한 통계적 정보(예: 평균, 표준편차)를 계산
  ```
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X = [[0, 15], [1, -10]]
  scaler.fit(X)  # 데이터의 평균과 표준편차를 계산
  print(scaler.mean_)  # 각 열의 평균
  print(scaler.scale_)  # 각 열의 표준편차
  ```

- transform(): `fit()`으로 계산된 통계적 정보를 기반으로 데이터를 변환.
  ```
  scaled_X = scaler.transform(X)  # 데이터를 표준화
  print(scaled_X)
  ```

- fit_transform(): 데이터의 통계적 특성을 계산(`fit`)하고, 이를 즉시 데이터에 적용(`transform`), 주로 훈련데이터에 사용

  ```
  scaled_X = scaler.fit_transform(X)  # fit()과 transform()을 동시에 수행
  print(scaled_X)
  ```

- fit()은 데이터를 분석하고, transform()은 데이터를 변환합니다. fit_transform()은 이 두 과정을 한 번에 수행합니다. 훈련 데이터에서는 fit_transform()을 사용하고, 테스트 데이터에서는 transform()만 사용해야 한다.

### **2.3. ColumnTransformer()**:
- 머신러닝 작업에서는 데이터의 각 특성(컬럼)이 서로 다른 형태와 스케일을 가질 수 있습니다. 
  - 수치형 데이터 : 표준화(Standardization)나 정규화(Normalization)가 필요.
  - 범주형 데이터 : 원-핫 인코딩(One-Hot Encoding)이나 레이블 인코딩(Label Encoding)이 필요.
  - 텍스트 데이터 : 토큰화(Tokenization), 벡터화(Vectorization) 등이 필요.
- 서로 다른 특성에 대해 다른 전처리를 적용하려면 `ColumnTransformer`를 사용합니다.
  ```
  import pandas as pd
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression

  # 샘플 데이터 생성
  data = {
      'age': [25, 45, 35, 50],
      'salary': [50000, 100000, 70000, 120000],
      'gender': ['male', 'female', 'female', 'male'],
      'city': ['New York', 'Paris', 'London', 'Tokyo']
  }
  df = pd.DataFrame(data)
  X = df.drop('city', axis=1)  # 입력 데이터
  y = df['city']              # 타겟 데이터

  # ColumnTransformer 정의
  preprocessor = ColumnTransformer( #각 튜플은 (이름, 변환기, 적용할 컬럼)으로 정의
      transformers=[
          ('num', StandardScaler(), ['age', 'salary']),  # 수치형 데이터: 표준화
          ('cat', OneHotEncoder(), ['gender'])           # 범주형 데이터: 원-핫 인코딩
      ]
  )

  # 파이프라인 생성
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', LogisticRegression())
  ])

  # 모델 학습
  pipeline.fit(X, y)

  # 새로운 데이터 예측
  new_data = pd.DataFrame({
      'age': [30],
      'salary': [80000],
      'gender': ['female']
  })
  print(pipeline.predict(new_data))
  ```


## **3. Pipelines: Chaining Pre-processors and Estimators (파이프라인)**
### **3.1. 핵심 개념**:
  - 여러 단계(전처리 + 모델 학습)를 하나의 파이프라인으로 연결하여 간단히 관리할 수 있습니다.
  - 파이프라인은 데이터 누출(Data Leakage)을 방지하는 데 유용합니다.
  - **예제 코드**:
    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe.fit(X_train, y_train)  # 파이프라인 학습
    print(accuracy_score(pipe.predict(X_test), y_test))  # 정확도 평가
    ```
  - **설명 포인트**:
    - 파이프라인은 전처리와 모델 학습을 하나의 객체로 묶어줍니다.
    - 데이터 누출(Data Leakage): 테스트 데이터 정보가 훈련 데이터에 유출되는 것을 방지합니다.

### **3.2. Pipeline()과 make_pipeline()**
**1. `Pipeline()`**
- **특징**:
  - 명시적으로 각 단계의 이름을 지정해야 합니다.
  - 각 단계는 `(이름, 객체)` 형태의 튜플로 정의됩니다.
- **사용 예시**:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression

  pipeline = Pipeline(steps=[
      ('scaler', StandardScaler()),          # 첫 번째 단계: 표준화
      ('classifier', LogisticRegression())  # 두 번째 단계: 분류 모델
  ])
  ```
  - 여기서 `'scaler'`와 `'classifier'`는 각 단계의 이름입니다.
  - 이름은 임의로 지정할 수 있지만, 중복되지 않아야 합니다.

**2. `make_pipeline()`**
  - **특징**:
    - 단계의 이름을 자동으로 생성합니다.
    - 단순히 객체를 순서대로 나열하면 됩니다.
  - **사용 예시**:
    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(
        StandardScaler(),          # 첫 번째 단계: 표준화
        LogisticRegression()      # 두 번째 단계: 분류 모델
    )
    ```
    - 여기서 각 단계의 이름은 자동으로 생성됩니다. 예를 들어:
      - `StandardScaler` → `'standardscaler'`
      - `LogisticRegression` → `'logisticregression'`

**3.차이점** 
- `Pipeline()`은 각 단계의 이름을 직접 지정할 수 있어 가독성과 참조가 쉽지만, 코드가 조금 더 복잡합니다. 반면에 `make_pipeline()`은 이름을 자동으로 생성해 주기 때문에 간단하고 빠르게 파이프라인을 만들 수 있습니다. 간단한 작업에는 `make_pipeline()`을, 복잡한 작업이나 디버깅이 필요한 경우에는 `Pipeline()`을 사용


## **4. Model Evaluation (모델 평가)**
- **핵심 개념**:
  - 모델의 성능은 반드시 테스트 데이터를 통해 평가해야 합니다.
  - 교차 검증(Cross-validation)은 모델의 일반화 성능을 평가하는 데 유용합니다.
- **예제 코드**:
  ```python
  from sklearn.datasets import make_regression
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_validate

  X, y = make_regression(n_samples=1000, random_state=0)
  lr = LinearRegression()
  result = cross_validate(lr, X, y)  # 5-fold 교차 검증
  print(result['test_score'])  # 각 폴드의 점수 출력
  ```
- **설명 포인트**:
  - `make_regression()`은  회귀(Regression) 문제를 위한 가상의 데이터셋을 생성
  - `cross_validate()`은 기본값이 5-fold
  - 교차 검증은 데이터를 여러 개의 폴드로 나누고, 각 폴드를 번갈아가며 테스트 세트로 사용합니다.
  - 이를 통해 모델의 안정성을 평가할 수 있습니다.


## **5. Automatic Parameter Searches (자동 하이퍼파라미터 탐색)**
### **5.1. 핵심 개념**:
  - 데이터 전처리 → 모델 학습 → 모델 평가 → 하이퍼파라미터 튜닝.
  - 모델의 성능은 하이퍼파라미터에 크게 의존합니다.
  - Scikit-learn은 자동으로 최적의 하이퍼파라미터를 찾는 도구를 제공합니다.
  - **예제 코드**:
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import randint

    param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                                n_iter=5,
                                param_distributions=param_distributions,
                                random_state=0)
    search.fit(X_train, y_train)  # 최적 파라미터 탐색
    print(search.best_params_)  # 최적 파라미터 출력
    print(search.score(X_test, y_test))  # 테스트 데이터 점수
    ```
  - **설명 포인트**:
    - `RandomizedSearchCV`: 무작위로 하이퍼파라미터 조합을 탐색합니다.
    - 최적의 파라미터를 찾으면 해당 설정으로 모델이 학습됩니다.

### **5.2. GridSearchCV**
- **특징**:
  - 모든 가능한 하이퍼파라미터 조합을 체계적으로 탐색합니다.
  - "완전 탐색(Exhaustive Search)" 방식으로, 지정된 파라미터 그리드의 모든 경우를 시도합니다.
  - 따라서 매우 정확한 결과를 제공하지만, 계산 비용이 매우 큽니다.

- **사용 예시**:
  ```python
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier

  # 하이퍼파라미터 그리드 정의
  param_grid = {
      'n_estimators': [10, 50, 100],
      'max_depth': [None, 10, 20]
  }

  # GridSearchCV 객체 생성
  grid_search = GridSearchCV(
      estimator=RandomForestClassifier(random_state=42),
      param_grid=param_grid,
      cv=5,  # 5-fold 교차 검증
      scoring='accuracy'
  )

  # 데이터 로드 및 학습
  X, y = ...  # 데이터셋 로드
  grid_search.fit(X, y)

  # 최적의 파라미터와 점수 출력
  print("Best Parameters:", grid_search.best_params_)
  print("Best Score:", grid_search.best_score_)
  ```

### **5.3. 어떤 경우에 사용할까?**
1. **`GridSearchCV`**:
   - 하이퍼파라미터 공간이 작고, 모든 조합을 체계적으로 탐색하고 싶을 때.
   - 예: `n_estimators=[10, 50, 100]`, `max_depth=[5, 10, 15]`처럼 제한된 범위

2. **`RandomizedSearchCV`**:
   - 하이퍼파라미터 공간이 크거나 연속적인 값(예: 실수 범위)을 포함할 때.
   - 예: `n_estimators=10~200`, `max_depth=None 또는 10~50`처럼 넓은 범위
