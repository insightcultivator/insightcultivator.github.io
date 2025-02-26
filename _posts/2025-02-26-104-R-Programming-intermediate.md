---
title: 10차시 2:R Programming(실습)
layout: single
classes: wide
categories:
  - R  
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 1. **Correlation**
-  `mtcars` 데이터셋을 분석하고 시각화하는 과정을 보여줍니다.

```R
install.packages('corrplot')
library(corrplot)

install.packages('lattice')
library(lattice)

a=mtcars
a
mcor2 = cor(mtcars$gear, mtcars$carb)
mcor2
xyplot(mtcars$gear~mtcars$carb, data=mtcars)

lm=plot(mtcars$gear, mtcars$carb)
abline(lm(mtcars$gear~mtcars$carb))

mcor=cor(mtcars)
mcor
round(mcor, 2)
corrplot(mcor)
plot(mtcars)


install.packages('ggplot2')
library(ggplot2)
qplot(gear, carb, data=mtcars)

cor(mtcars$wt, mtcars$mpg)
qplot(wt, mpg, data=mtcars, color=factor(carb))
```



1. **패키지 설치 및 로드**:  
   - `corrplot`, `lattice`, `ggplot2` 패키지를 설치하고 불러옵니다. 이들은 데이터 시각화와 상관관계 분석에 사용됩니다.

2. **데이터 탐색**:  
   - `mtcars` 데이터셋을 변수 `a`에 저장하고 확인합니다.

3. **상관관계 계산**:  
   - `cor()` 함수로 `gear`와 `carb` 간 상관계수(`mcor2`)를 계산하고, 전체 변수 간 상관행렬(`mcor`)을 구한 뒤 소수점 2자리로 반올림합니다.

4. **시각화**:  
   - `xyplot()` (lattice): `gear`와 `carb` 간 산점도를 그림.
   - `plot()`: `gear`와 `carb`의 산점도와 선형 회귀선(`abline`)을 추가.
   - `corrplot()`: 전체 변수 간 상관행렬을 시각화.
   - `qplot()` (ggplot2): `gear`와 `carb`, 그리고 `wt`와 `mpg` 간 산점도를 그리며, `carb`를 색상으로 구분한 그래프도 생성.



## 2. **Regression**
- 연차와 연봉 간의 상관관계를 탐색하고, 연차가 연봉에 미치는 영향을 선형 회귀를 통해 분석

```R
year = c(26, 16, 20, 7, 22, 15, 29, 28, 17, 3, 1, 16, 19, 13, 27, 4, 30, 8, 3, 12)
annual_salary=c(1267, 887, 1022, 511, 1193, 795, 1713, 1477, 991, 455, 324, 944, 1232, 808, 1296, 486, 1516, 565, 299, 830)
Data = data.frame(year, annual_salary)

summary(Data)

plot(year,annual_salary)
cor(year,annual_salary)

LS = lm(annual_salary~year, data=Data)
summary(LS)

```

1. **데이터 구성**: 
   - `year`: 근무 연수 데이터 (20개 값, 예: 26, 16, 20 등).
   - `annual_salary`: 연봉 데이터 (20개 값, 예: 1267, 887, 1022 등).
   - 두 변수를 `Data`라는 데이터프레임으로 묶음.

2. **기초 분석**:
   - `summary(Data)`: 데이터의 기초 통계량(최소값, 최대값, 평균 등)을 확인.
   - `plot(year, annual_salary)`: 연차와 연봉 간의 산점도를 그림.
   - `cor(year, annual_salary)`: 두 변수 간 상관계수를 계산해 관계 강도를 확인.

3. **선형 회귀 분석**:
   - `lm(annual_salary~year, data=Data)`: 연차를 독립변수로, 연봉을 종속변수로 하는 선형 회귀 모델(`LS`)을 생성.
   - `summary(LS)`: 회귀 분석 결과(기울기, 절편, 유의미성 등)를 요약.


## 3. **ANOVA**
- `anorexia` 데이터셋을 분석하는 과정을 보여줍니다. 이 데이터셋은 `MASS` 패키지에 포함되어 있으며, 식이장애 환자의 치료 전후 체중 변화를 연구한 데이터를 다룬다.

```R
data(anorexia, package = 'MASS')
anorexia

install.packages('car')
library(car)
leveneTest(Postwt~Treat, data = anorexia)

out1 = aov(Postwt~Treat, data=anorexia)
out1
summary(out1)

out2 = anova(lm(Postwt~Treat, data = anorexia))
out2

out3 = oneway.test(Postwt~Treat, data = anorexia)
out3
```

1. **데이터 로드**: `anorexia` 데이터셋을 `MASS` 패키지에서 불러옵니다. 이 데이터셋에는 치료 방식(Treat)과 치료 후 체중(Postwt) 등의 변수가 포함되어 있습니다.

2. **Levene 검정**: `car` 패키지의 `leveneTest` 함수를 사용해 치료 그룹(Treat) 간 체중(Postwt)의 분산이 동일한지 확인합니다. 이는 이후 분석의 전제 조건을 검토하는 단계입니다.

3. **일원분산분석(ANOVA)**: 세 가지 방법으로 치료 방식(Treat)이 치료 후 체중(Postwt)에 미치는 영향을 분석합니다.
   - `aov`: 기본적인 분산분석을 수행하고 결과를 `out1`에 저장합니다.
   - `anova`와 `lm`: 선형 모델을 적용한 후 분산분석을 수행하며, 결과를 `out2`에 저장합니다.
   - `oneway.test`: 분산이 다를 가능성을 고려한 일원분산분석을 수행하며, 결과를 `out3`에 저장합니다.

4. **결과 출력**: 각 분석의 요약(`summary(out1)`) 또는 결과를 직접 확인(`out2`, `out3`)하여 치료 방식에 따른 체중 차이가 통계적으로 유의미한지 평가합니다.




## 4. **PCA**
- 학생 16명의 네 과목(국어, 영어, 수학, 과학) 점수를 데이터 프레임으로 구성한 뒤, 주성분 분석(PCA, Principal Component Analysis)을 수행하는 과정을 보여줍니다.

```R
x1=c(26,46,57,36,57,26,58,37,36,56,78,95,88,90,52,56)
x2=c(35,74,73,73,62,22,67,34,22,42,65,88,90,85,46,66)
x3=c(35,76,38,69,25,25,87,79,36,26,22,36,58,36,25,44)
x4=c(45,89,54,55,33,45,67,89,47,36,40,56,68,45,37,56)

score=cbind(x1,x2,x3,x4)
score
colnames(score)=c("국어", "영어", "수학", "과학")
rownames(score)=1:16
head(score)
result=prcomp(score)
result
summary(result)
```

1. **데이터 생성**: `x1`, `x2`, `x3`, `x4`는 각각 국어, 영어, 수학, 과학 점수를 나타내는 벡터로, 16명의 학생 데이터를 포함
2. **데이터 결합**: `cbind`를 사용해 네 벡터를 열로 결합하여 `score`라는 데이터 프레임으로
3. **열 및 행 이름 지정**: 열 이름은 과목명(국어, 영어, 수학, 과학)으로, 행 이름은 1부터 16까지 번호로 지정
4. **데이터 확인**: `head(score)`로 데이터의 처음 몇 행을 확인
5. **주성분 분석 수행**: `prcomp(score)`를 통해 PCA를 실행하고, 결과는 `result`에 저장
6. **결과 요약**: `summary(result)`로 주성분 분석 결과를 요약해 출력(예: 각 주성분의 설명 분산 비율 등).


## 5. **Logistic Regression**
- 용량(dose)에 따른 반응(response)의 확률을 분석하고 예측하는 로지스틱 회귀 모델

```R

dose = c(1,1,2,2,3,3)
response = c(0,1,0,1,0,1)
count = c(7,3,5,5,2,8)
toxic = data.frame(dose, response, count)
toxic

out = glm(response~dose, weights = count, family = binomial, data=toxic)
out
summary(out)

plot(response~dose, data=toxic, type='n',main='Predicted Probability of Response')
curve(predict(out, data.frame(dose=x), type='resp'), add=TRUE)
```

1. **데이터 준비**:  
   - `dose`(용량), `response`(반응 여부: 0 또는 1), `count`(각 경우의 빈도)라는 세 변수를 정의하고, 이를 `toxic`이라는 데이터프레임으로 구성합니다.
2. **모델 생성**:  
   - `glm()` 함수를 사용해 로지스틱 회귀 모델을 만듭니다.  
   - `response~dose`: 반응을 용량으로 설명하는 모델.  
   - `weights = count`: 각 데이터 포인트의 빈도를 반영.  
   - `family = binomial`: 이항 분포를 가정(로지스틱 회귀에 적합).  
3. **결과 확인**:  
   - `out`은 모델 객체이고, `summary(out)`로 모델의 통계적 요약(계수, 유의성 등)을 확인.
4. **시각화**:  
   - `plot()`과 `curve()`를 사용해 용량에 따른 예측 반응 확률을 그래프로 나타냅니다.  
   - `type='n'`은 빈 그래프를 먼저 그리고, `predict()`로 계산된 확률 곡선을 추가.



## 6. **Prediction Analytics**
- `iris` 데이터셋을 기반으로 의사결정나무를 만들고, 이를 시각적으로 표현하는 과정을 수행

```R
install.packages('rpart')
library(rpart)

formula = Species~.
iris.df = rpart(formula, data=iris)
iris.df
plot(iris.df)
text(iris.df, use.n = T, cex=0.5)
post(iris.df, file="")

install.packages("rpart")
library(rpart) #의사결정나무 만들기  
#petal(꽃잎) length(길이) width(넓이)
#sepal(꽃받침)
head(iris) #setosa,
x11(800,600)   # 그래픽 윈도우 장치 열기                                                                  
formula = Species ~ .
iris.df = rpart(formula, data=iris,method = "class")
iris.df  
plot(iris.df ) 
text(iris.df, use.n=T, cex=0.7) 
post(iris.df, file="")

```

1. **`rpart` 패키지 설치 및 로드**: 의사결정나무 모델을 만들기 위해 `rpart` 패키지를 설치하고 불러옵니다.
2. **데이터 및 모델 설정**: `iris` 데이터셋을 사용하며, `Species`를 종속 변수로, 나머지 변수(`.`)를 독립 변수로 설정하여 의사결정나무 모델(`iris.df`)을 생성합니다.
3. **모델 시각화**: 
   - `plot()`으로 의사결정나무를 그리고, 
   - `text()`로 노드에 레이블을 추가하며(`use.n=T`로 샘플 수 표시, `cex`로 텍스트 크기 조정),
   - `post()`로 결과를 파일로 출력(파일명 미지정 시 기본값 사용).
4. **추가 요소**: `head(iris)`로 데이터 일부를 확인하고, `x11()`로 그래픽 창을 엽니다.




## 7. **Time Series**
- 대한민국의 1인당 GDP와 전체 GDP 데이터를 분석하고, 이를 바탕으로 2018년부터 2021년까지의 국민총생산을 예측하는 과정

```R
# 국민총생산 예측 2018~2021
install.packages("WDI")
library(WDI)
gdp <- WDI(country="KR",
           indicator=c("NY.GDP.PCAP.CD", "NY.GDP.MKTP.CD"),
           start=1960, end=2017)
head(gdp)
names(gdp) <- c("Country","iso2c","iso3c", "Year", "PerCapGDP", "GDP")
head(gdp)

kr=gdp$PerCapGDP[gdp$Country=="Korea, Rep."]
kr=ts(kr, start=min(gdp$Year), end=max(gdp$Year))
kr

install.packages("forecast")
library(forecast)
krts=auto.arima(x=kr) #시계열 분석 예측 
krts
Forecasts=forecast(object=krts, h=5)
Forecasts
plot(Forecasts)
```

1. **데이터 수집**: `WDI` 패키지를 이용해 세계은행(World Bank)에서 제공하는 대한민국의 1960년부터 2017년까지의 1인당 GDP(`NY.GDP.PCAP.CD`)와 전체 GDP(`NY.GDP.MKTP.CD`) 데이터를 가져옵니다.
2. **데이터 정리**: 가져온 데이터를 정리하여 열 이름을 "Country", "Year", "PerCapGDP", "GDP" 등으로 변경하고, 대한민국 데이터만 추출해 시계열 데이터(`ts`)로 변환합니다.
3. **예측 모델 생성**: `forecast` 패키지의 `auto.arima` 함수를 사용해 시계열 데이터를 분석하고, ARIMA 모델을 자동으로 적합시킵니다.
4. **예측 수행**: 적합된 모델을 바탕으로 향후 5년(2018~2022년)의 GDP를 예측하고, 결과를 `Forecasts` 객체에 저장합니다.
5. **시각화**: 예측 결과를 그래프로 표시합니다.


## 8. **Clustering**
- iris 데이터셋을 이용해 K-평균 군집화(K-means clustering)를 수행하는 과정을 보여줍니다. 

```R

data(iris)
a=iris
a
a$Species=NULL #사전 정보 없이
a
kc=kmeans(a,3)
kc
summary(kc)
table(iris$Species,kc$cluster)
plot(a[c("Sepal.Length","Sepal.Width")],col=kc$cluster)
```

1. **데이터 준비**: iris 데이터셋을 불러와 변수 `a`에 저장하고, `Species` 열(꽃의 종류)을 제거합니다. 즉, 사전 정보 없이 꽃의 특성만으로 군집화를 진행합니다.
2. **군집화 수행**: `kmeans` 함수를 사용해 데이터를 3개의 군집으로 나눕니다. 결과는 `kc`에 저장됩니다.
3. **결과 확인**: 
   - `summary(kc)`로 군집화 결과 요약을 확인하고, 
   - `table` 함수로 실제 종(Species)과 군집 결과를 비교합니다.
4. **시각화**: `Sepal.Length`(꽃받침 길이)와 `Sepal.Width`(꽃받침 너비)를 기준으로 산점도를 그리고, 각 점을 군집에 따라 색으로 구분해 표시합니다.

## 9. **Derived Variable**
-  항공편 데이터를 불러와 구조를 확인한 뒤, 도착 지연과 출발 지연의 차이를 계산하여 새로운 열을 추가하는 작업을 수행

```R

install.packages("hflights")
install.packages("dplyr")
library(hflights)
library(dplyr)
str(hflights)  
colnames(hflights) 
hflight_df <- as_tibble(hflights) 
hflight_df
aa<-mutate(hflight_df , gain=ArrDelay - DepDelay) 
aa
```

1. **패키지 설치 및 불러오기**:  
   - `hflights` 패키지(항공편 데이터 포함)와 `dplyr` 패키지(데이터 조작 도구)를 설치하고 불러옵니다.

2. **데이터 확인**:  
   - `str(hflights)`로 데이터 구조를 확인하고, `colnames(hflights)`로 열 이름을 출력합니다.

3. **데이터 변환**:  
   - `hflights` 데이터를 `tibble` 형식(더 현대적인 데이터프레임)으로 변환하여 `hflight_df`에 저장합니다.

4. **새로운 열 추가**:  
   - `mutate` 함수를 사용해 `ArrDelay`(도착 지연 시간)에서 `DepDelay`(출발 지연 시간)를 뺀 값으로 `gain`이라는 새 열을 생성하고, 결과를 `aa` 변수에 저장합니다.


## 10. **Ensemble Analytics**
- iris 데이터셋을 기반으로 꽃의 종을 예측하는 RandomForest(앙상블의 bagging)분류 모델을 만들고 평가하는 과정

```R
install.packages("randomForest")
library(randomForest)
head(iris)
idx=sample(2,nrow(iris),replace=T,prob=c(0.7,0.3))
idx
trainData=iris[idx==1,]
trainData
testData=iris[idx==2,]
testData
model=randomForest(Species~.,data=trainData,ntree=100,proximity=T)
table(trainData$Species, predict(model))
importance(model)
model
```

1. **패키지 설치 및 로드**: `randomForest` 패키지를 설치하고 불러옵니다.
2. **데이터 준비**: `iris` 데이터셋을 사용하며, 데이터를 무작위로 훈련용(`trainData`)과 테스트용(`testData`)으로 나눕니다. 나눌 때 70%는 훈련, 30%는 테스트로 설정합니다.
3. **모델 학습**: 훈련 데이터를 사용해 `randomForest` 모델을 생성합니다. `Species`를 예측 대상으로, 나머지 변수를 입력 변수로 사용하며, 트리 개수는 100개로 설정합니다.
4. **결과 확인**: 훈련 데이터에 대한 예측 결과와 실제 값의 비교표를 만들고, 각 변수의 중요도를 확인


## 11. **Prediction Error**
- `cars` 데이터를 활용해 속도와 거리 간의 선형 관계를 분석하고, 모델의 적합성과 가정을 검토하는 전형적인 회귀 분석 워크플로우를 보여줍니다.

```R
head(cars,3)
str(cars)
?cars
a=lm(dist~speed, cars)
a
coef(a)
fitted(a)[1:4]
residuals(a)[1:4]
confint(a)
deviance(a)
predict(a,newdata=data.frame(speed=4))
coef(a)[1]+coef(a)[2]*4
predict(a,newdata=data.frame(speed=4),interval="confidence")
predict(a,newdata=data.frame(speed=4),interval="prediction")
summary(a)
par(mfrow=c(2,2))
plot(a)

res=residuals(a)
shapiro.test(res)
install.packages("lmtest")
library(lmtest)
dwtest(a)
```

1. **데이터 확인**:  
   - `head(cars, 3)`: `cars` 데이터의 처음 3행을 확인.  
   - `str(cars)`: 데이터 구조를 파악.  
   - `?cars`: `cars` 데이터셋에 대한 도움말 호출.

2. **선형 회귀 모델 생성**:  
   - `a = lm(dist ~ speed, cars)`: 속도(`speed`)를 독립변수로, 거리(`dist`)를 종속변수로 하는 선형 회귀 모델 생성.  
   - `coef(a)`: 회귀 계수(절편과 기울기) 확인.  

3. **모델 결과 분석**:  
   - `fitted(a)[1:4]`: 처음 4개 관측값의 예측값 계산.  
   - `residuals(a)[1:4]`: 처음 4개 관측값의 잔차 확인.  
   - `confint(a)`: 회귀 계수의 신뢰구간 계산.  
   - `deviance(a)`: 모델의 편차 계산.  

4. **예측**:  
   - `predict(a, newdata=data.frame(speed=4))`: 속도가 4일 때 거리 예측.  
   - `coef(a)[1] + coef(a)[2]*4`: 수동으로 동일한 예측값 계산.  
   - `predict(...interval="confidence")`: 신뢰구간 포함 예측.  
   - `predict(...interval="prediction")`: 예측구간 포함 예측.

5. **모델 요약 및 시각화**:  
   - `summary(a)`: 모델 요약(계수, R², p값 등).  
   - `par(mfrow=c(2,2)); plot(a)`: 회귀 진단을 위한 4개의 플롯 생성(잔차 분석 등).

6. **잔차 진단**:  
   - `shapiro.test(residuals(a))`: 잔차의 정규성 검정(Shapiro-Wilk 테스트).  
   - `dwtest(a)`: 잔차의 독립성 검정(Durbin-Watson 테스트, `lmtest` 패키지 필요).



## 12. **K-Flod**
- 아이리스(iris) 데이터셋을 기반으로 의사결정나무 모델(ctree)을 생성하고, 교차 검증(cross-validation)을 통해 모델의 성능을 평가하는 과정


```R
install.packages("party")
library(party)
install.packages("cvTools")
library(cvTools)
head(iris)
str(iris)
cross=cvFolds(nrow(iris),K=3)
str(cross)
cross
cross$which
cross$subsets
k=1:3
acc=numeric()
cnt=1
for(i in k){
  data_index=cross$subsets[cross$which==i,1]
  test=iris[data_index,]
  formula=Species~.
  train=iris[-data_index,] # 전체갯수에서 test 데이터를 빼야하므로 -를 붙임 
  model=ctree(formula, data=train)
  pred=predict(model, test)
  t=table(pred, test$Species)
  print(t)
  acc[cnt]=(t[1,1]+t[2,2]+t[3,3])/sum(t)
  cnt=cnt+1
}
nrow(test) # 50개 test
nrow(train) #100개 train
acc
mean(acc)
```

1. **패키지 설치 및 로드**: `party`와 `cvTools` 패키지를 설치하고 불러옵니다. `party`는 의사결정나무 모델을 만들기 위해, `cvTools`는 교차 검증을 수행하기 위해 사용됩니다.

2. **데이터 준비**: 아이리스 데이터셋을 확인하고, 3겹 교차 검증(3-fold cross-validation)을 설정합니다. 이를 위해 `cvFolds` 함수를 사용해 데이터 인덱스를 3개의 폴드로 나눕니다.

3. **모델 학습 및 예측**:
   - 데이터를 학습(train)과 테스트(test) 세트로 나눕니다.
   - 학습 데이터를 이용해 Species(붓꽃 종)를 예측하는 의사결정나무 모델을 생성합니다.
   - 테스트 데이터를 사용해 예측을 수행하고, 실제 값과 예측 값을 비교한 혼동 행렬(table)을 출력합니다.

4. **정확도 계산**: 각 폴드별로 모델의 정확도를 계산하여 `acc` 벡터에 저장합니다. 정확도는 혼동 행렬에서 대각선 값(정확히 예측된 경우)을 전체 합으로 나눈 값입니다.

5. **결과 확인**: 테스트 데이터(50개)와 학습 데이터(100개)의 크기를 확인하고, 각 폴드의 정확도(`acc`)와 평균 정확도(`mean(acc)`)를 계산합니다.



## 13. **Confusion Matrix**
- iris 데이터셋을 활용해 의사결정나무로 붓꽃 종을 분류하고, 모델의 성능을 훈련 및 테스트 데이터로 평가하는 과정

```R
head(iris, 10)
summary(iris)
install.packages("party")
install.packages("caret")
install.packages("e1071")
library(party)
library(caret)
library(e1071)
sp=sample(2,nrow(iris),replace=TRUE, prob=c(0.7,0.3))
trainData=iris[sp==1,]
testData=iris[sp==2,]
myFomula=Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width
iris_ctree=ctree(myFomula, data=trainData)
myFomula=Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width
iris_ctree=ctree(myFomula, data=trainData)
table(predict(iris_ctree),
      trainData$Species)
confusionMatrix(predict(iris_ctree),
                trainData$Species)
plot(iris_ctree)
testPred=predict(iris_ctree,newdata=testData)
table(testPred, testData$Species)
confusionMatrix(testPred, 
                testData$Species)
```

1. **데이터 확인**: iris 데이터셋의 처음 10개 행을 확인하고(`head`), 전체 데이터의 요약 통계를 출력(`summary`)합니다.

2. **패키지 설치 및 로드**: 의사결정나무 모델(`party`), 모델 평가(`caret`), 기계학습 알고리즘(`e1071`)을 사용하기 위해 필요한 패키지를 설치하고 불러옵니다.

3. **데이터 분할**: iris 데이터를 무작위로 훈련 데이터(70%)와 테스트 데이터(30%)로 나눕니다.

4. **모델 학습**: 훈련 데이터를 사용해 `Species`(붓꽃 종)를 예측하는 의사결정나무 모델(`ctree`)을 만들고, 입력 변수로 꽃받침 길이/너비와 꽃잎 길이/너비를 사용합니다.

5. **모델 평가**: 
   - 훈련 데이터에 대한 예측 결과와 실제 값을 비교해 혼동 행렬(`confusionMatrix`)을 생성합니다.
   - 모델 구조를 시각화(`plot`)합니다.
   - 테스트 데이터로 예측을 수행하고, 마찬가지로 혼동 행렬을 통해 성능을 평가합니다.

## 14. **ROC**
- 의사결정나무(decision tree) 모델을 구축하고 평가

```R
install.packages("rpart")

library(rpart)
?DMwR
data(hacide)
str(hacide.train)
table(hacide.train$cls)
prop.table(table(hacide.train$cls))
tree=rpart(cls ~.,data=hacide.train)
tree
pred.tree=predict(tree,newdata=hacide.test)
pred.tree
accuracy.meas(hacide.test$cls,pred.tree[,2])
roc.curve(hacide.test$cls, pred.tree[,2],plotit=T)
```

1. **패키지 설치 및 로드**: `rpart` 패키지를 설치하고 불러옵니다. 이 패키지는 의사결정나무 모델을 구현하는 데 사용됩니다.
2. **데이터 준비**: `hacide`라는 데이터셋을 사용하며, `hacide.train` 데이터의 구조와 클래스 분포를 확인합니다.
3. **모델 학습**: `rpart` 함수를 통해 `cls`라는 목표 변수를 나머지 변수들로 예측하는 의사결정나무 모델(`tree`)을 학습시킵니다.
4. **예측**: 학습된 모델을 활용해 테스트 데이터(`hacide.test`)에 대한 예측값(`pred.tree`)을 생성합니다.
5. **모델 평가**: 예측 정확도를 측정하고 ROC 곡선을 그려 모델의 성능을 시각적으로 평가합니다.



## 15. **Model Evaluation**
- 타이타닉 데이터셋을 전처리하고, 로지스틱 회귀분석을 통해 생존 여부를 예측한 뒤 ROC 곡선과 AUC를 계산하는 과정

```R

# 데이터 전처리와 로지스틱 회귀분석에서 ROC곡선 패키지
install.packages('functional')
install.packages('ROCR')
library(functional)
library(ROCR)
Titanic

# 테이블 형태의 자료를 행과 열로 만들어진 리스트로 변환한다.
pivot.titanic <- as.data.frame(Titanic)
pivot.titanic
#모든 코드에서 컬럼을 직접 접근 
attach(pivot.titanic)
nrow(Titanic)
nrow(pivot.titanic)

# 데이터셋을 피벗 요약 테이블 형태에서 1명당 1행을 가진 형태로 변형
titanic.class <- c()
titanic.sex <- c()
titanic.age <- c()
titanic.survived <- c()

for(i in 1:nrow(pivot.titanic)){ 
  n.rep <- functional::Curry(rep,times=Freq[i])
  titanic.class <- append(titanic.class, n.rep(as.character(Class[i])))
  titanic.sex <- append(titanic.sex, n.rep(as.character(Sex[i])))
  titanic.age <- append(titanic.age, n.rep(as.character(Age[i])))
  titanic.survived <- append(titanic.survived, n.rep(as.character(Survived[i])))
}
detach(pivot.titanic)
titanic.class
titanic.sex
titanic.survived
titanic = 
  data.frame(
    Idx=1:length(titanic.class),
    Class=titanic.class, 
    Sex=titanic.sex, 
    Age=titanic.age, 
    Survived=titanic.survived)
head(titanic)
summary(titanic)

#각 클래스들의 비율을 맞추어 표본을 뽑기 위해 층화추출을 적용
#  모집단을 먼저 중복되지 않도록 층으로 나눈 다음 각 층에서 표본을 추출하는 방법이다. 층을 나눌 때 
#  층내는 동질적(homogeneous), 층간은 이질적(heterogeneous) 특성을
#  갖도록 한다.
sampling.info <- aggregate(Freq ~ Class + Sex + Age, pivot.titanic, sum)
sampling.info
test.ratio <- 0.7

test.idx <- c()
for(i in 1:nrow(sampling.info)){
  target.row <- sampling.info[i,]
  key <- target.row[1:3]
  target.rows.idx <- merge(x=key, y=titanic, by=c('Class','Sex','Age'))$Idx
  test.idx <- append(test.idx, sample(target.rows.idx, target.row$Freq * test.ratio))
}
test <- titanic[test.idx, ]
train.idx <- setdiff(titanic$Idx, test.idx)
train <- titanic[train.idx, ]
nrow(train)
nrow(test)
nrow(titanic)
summary(train)
summary(test)

#로지스틱 회귀분석 모형 적용
model <- glm(as.factor(Survived) ~ Class + Sex + Age, family='binomial', data=train)
summary(model)

# 스코어 테이블 출력과 AUC 값 출력
require(ROCR)
prob <- predict(model, 
                newdata=test, type='response')
prob
labeled.score <- 
  merge(titanic, 
        data.frame(Idx=
                     as.integer(names(prob)), 
                   score=prob), 
        by=c('Idx'))
write.csv( labeled.score[with(labeled.score, order(-score)), ], 
           file='titanic_logistic_score.csv')
pred <- prediction(prob, test$Survived)
roc <- performance(pred, measure = 'tpr', x.measure = 'fpr')
plot(roc, col='red')
legend('bottomright', c('base','logistic'), col=1:2, lty=2:1)
abline(a=0, b=1, lty='dotted')
auc <- performance(pred, measure = 'auc')
auc <- auc@y.values[[1]]
auc
```

1. **데이터 전처리**:  
   - 타이타닉 데이터를 피벗 테이블 형태에서 개별 행(1명당 1행)으로 변환합니다.  
   - `Class`, `Sex`, `Age`, `Survived` 열을 생성하고, 빈도(`Freq`)를 기반으로 데이터를 확장합니다.

2. **층화 추출**:  
   - 데이터를 `Class`, `Sex`, `Age` 기준으로 층화하고, 70%를 학습용(`train`), 나머지를 테스트용(`test`)으로 나눕니다.

3. **로지스틱 회귀분석**:  
   - `Survived`를 종속 변수로, `Class`, `Sex`, `Age`를 독립 변수로 사용해 모델을 학습합니다.

4. **예측 및 평가**:  
   - 테스트 데이터로 생존 확률을 예측하고, ROC 곡선을 그리며 AUC 값을 계산합니다.  
   - 결과는 `titanic_logistic_score.csv` 파일로 저장됩니다.





