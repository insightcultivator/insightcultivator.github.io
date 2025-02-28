---
title: 12차시 1:Computer Vision(Basic)
layout: single
classes: wide
categories:
  - Computer Vision
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 1. **이미지 데이터 이해**

**1. 이미지 표현 방식 심화**
* 이미지는 어떻게 컴퓨터가 이해할 수 있는 형태로 저장될까요? 기본적으로 RGB와 Grayscale 외에도 다양한 색 공간(Color Space)이 존재합니다. 이를 통해 우리는 이미지를 더 효과적으로 처리할 수 있습니다.

    - **HSV(Hue, Saturation, Value):**
    HSV는 인간의 시각 체계에 맞춰진 색상 표현 방식입니다.  
        - **Hue(색상):** 색의 종류 (예: 빨강, 파랑)
        - **Saturation(채도):** 색의 선명도
        - **Value(명도):** 색의 밝기  

    - **YCrCb:**
        - YCrCb는 영상 압축 및 방송 표준에서 사용되는 색 공간입니다.  
        - **Y:** 밝기 정보 (Luminance)
        - **Cr, Cb:** 색차 정보 (Chrominance)

    - **이미지 양자화:**
        - 양자화는 이미지의 색상 수를 줄여 저장 공간을 최적화하는 기법입니다. 
        - 예를 들어, 24비트 RGB 이미지를 8비트로 변환하면 파일 크기를 크게 줄일 수 있지만, 시각적 품질이 저하될 수 있습니다.

    - **실습: OpenCV를 활용한 색 공간 변환**

        ```python
        import cv2
        import matplotlib.pyplot as plt

        # 이미지 로드
        image = cv2.imread('example.jpg')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드함

        # RGB -> HSV 변환
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # 결과 시각화
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(image_rgb), plt.title('Original (RGB)')
        plt.subplot(1, 2, 2), plt.imshow(image_hsv[:, :, 0], cmap='hsv'), plt.title('HSV (Hue Channel)')
        plt.show()
        ```
    ![hue.png](/assets/images/HSV.png)


**2. 이미지 데이터의 저장 형식 심층 분석**
- JPEG, PNG, GIF 등 다양한 이미지 파일 형식이 존재하며, 각 형식은 특정 용도에 적합합니다.  
    - **JPEG:** 손실 압축, 사진에 적합  
    - **PNG:** 무손실 압축, 투명도 지원  
    - **GIF:** 애니메이션 지원  

- **실습: PIL을 활용한 이미지 형식 변환**

    ```python
    from PIL import Image

    # 이미지 로드
    img = Image.open('example.jpg')

    # JPEG -> PNG 변환
    img.save('example_converted.png', 'PNG')
    print("이미지 형식 변환 완료!")
    ```

## 2. **데이터셋: 컴퓨터 비전의 핵심 자원**
* 컴퓨터 비전 모델을 학습하기 위해서는 대규모 데이터셋이 필요합니다. 여기서는 몇 가지 중요한 공개 데이터셋을 소개합니다.

    - **ImageNet**
        - ImageNet은 1,000개 이상의 클래스로 구성된 대규모 이미지 분류 데이터셋입니다. 이 데이터셋은 딥러닝 연구의 기반이 되었으며, ResNet, VGG 등의 모델 개발에 큰 역할을 했습니다.

    - **COCO**
        - COCO(Common Objects in Context)는 객체 탐지, 세그멘테이션, 이미지 캡셔닝 등 다양한 작업을 위한 데이터셋입니다. 각 이미지에는 객체 경계 상자(Bounding Box)와 세부 레이블이 포함되어 있습니다.

    - **Cityscapes**
        - Cityscapes는 자율 주행 분야에서 사용되는 도시 환경 데이터셋으로, 차선 인식, 보행자 탐지 등에 활용됩니다.

- **실습: PyTorch를 활용한 데이터셋 로딩**
    
    ```python
    import torch
    from torchvision import datasets, transforms

    # 데이터 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # CIFAR-10 데이터셋 로딩
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 데이터 시각화
    images, labels = next(iter(train_loader))
    print(f"배치 크기: {images.shape}, 레이블: {labels}")
    ```


## 3. **이미지 전처리 (Image Preprocessing)**  
- **실습: 노이즈 제거 및 밝기 조정(OpenCV)**  

```python
import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('noisy_image.jpg')

# 노이즈 제거 (블러 처리)
denoised_image = cv2.GaussianBlur(image, (5, 5), 0)

# 밝기 조정
brightness_adjusted = cv2.convertScaleAbs(denoised_image, alpha=1.5, beta=30)

# 결과 표시
cv2.imshow('Original', image)
cv2.imshow('Denoised', denoised_image)
cv2.imshow('Brightness Adjusted', brightness_adjusted)
cv2.waitKey(0)
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/denoised.png" alt="denoised image" width="500">

- **과제**: 자신의 사진을 가져와 노이즈를 줄이고 밝기를 조정해보기.


## 4. **기본 이미지 분석 (Low-Level Vision)**  
- **실습: 에지 감지(OpenCV)**  

```python
import cv2

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('object.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 에지 감지
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# 결과 표시
cv2.imshow('Original', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/edge_profile.png" alt="denoised image" width="500">

- **과제**: 다양한 임계값(threshold)을 실험하며 에지 감지 결과를 비교해보기.



## 5. **패턴 인식 및 분류 (Pattern Recognition & Classification)**  
- **실습 예제: 손글씨 숫자 인식 (MNIST 데이터셋,TensorFlow/Keras)**  

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 정의
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")
```

- **과제**: 자신이 직접 쓴 숫자 이미지를 모델에 입력하여 분류 결과 확인하기.



## 6. **객체 탐지 (Object Detection)**  
- **실습 예제: YOLO를 사용한 객체 탐지**  

```python
import cv2

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 이미지 로드 및 전처리
image = cv2.imread('street.jpg')
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 객체 탐지 결과 시각화
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/yolo_street.png" alt="yolo street" width="400">

- **과제**: 다양한 이미지에서 객체 탐지 결과를 확인하고 경계 상자의 정확도를 평가하기.



## 7. **시맨틱 세분화 (Semantic Segmentation)**  
- **실습: U-Net을 사용한 도로/보행자 세분화**  

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Pretrained U-Net 모델 로드
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# 이미지 로드 및 전처리
image = Image.open('road.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# 추론
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0)

# 결과 시각화
import matplotlib.pyplot as plt
plt.imshow(output_predictions.byte().cpu().numpy())
plt.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/semantic_road.png" alt="semantic road" width="400">

- **과제**: 도로 이미지를 가져와 도로와 보행자를 구분하는 세그멘테이션 결과 확인하기.

## 8. **도메인 적응: 새로운 환경에서의 적용**
- 도메인 적응(Domain Adaptation)은 한 환경에서 학습된 모델을 다른 환경에서도 잘 작동하도록 조정하는 기법입니다. 예를 들어, 사전 학습된 ImageNet 모델을 의료 이미지 데이터에 적용할 때 유용합니다.

- **Transfer Learning**
    - Transfer Learning은 사전 학습된 모델의 특징을 활용하여 새로운 작업에 적용하는 방법입니다. Fine-tuning과 Feature Extraction이 주요 전략입니다.

- **실습: 사전 학습된 ResNet 모델을 활용한 전이 학습**

```python
import torch
import torchvision.models as models
from torchvision import transforms

# 사전 학습된 ResNet 모델 로드
model = models.resnet18(pretrained=True)

# 마지막 레이어 수정 (새로운 클래스 수에 맞춤)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 10개 클래스로 변경

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```


## 9. **3D 이해 및 재구성 (3D Vision)**  
- **실습 예제: Stereo Matching을 사용한 깊이 맵 생성(OpenCV)**  
- 간단한 테스트용 샘플
    - Middlebury 데이터셋에서 Tsukuba 이미지 쌍을 추천합니다. 이 데이터셋은 간단하고 작은 크기의 이미지(left.png, right.png)를 제공하며, 실습에 적합합니다.
    - 구체적인 링크: [Tsukuba Stereo Pair](https://vision.middlebury.edu/stereo/data/scenes2001/) (여기서 "Tsukuba" 이미지 다운로드).
    - scene1.row3.col2.ppm (왼쪽 시점)을 left.jpg로 저장.
    - scene1.row3.col4.ppm (오른쪽 시점)을 right.jpg로 저장
    
```python
import cv2
import numpy as np

# 스테레오 이미지 로드
imgL = cv2.imread("left.jpg", 0)
imgR = cv2.imread("right.jpg", 0)

# StereoBM 알고리즘 설정
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=13)
disparity = stereo.compute(imgL, imgR)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(imgL), plt.title('Left')
plt.subplot(1, 3, 2), plt.imshow(imgR), plt.title('Right')
plt.subplot(1, 3, 3), plt.imshow(disparity, "gray")
plt.show()
```

![stereo BM](/assets/images/stereo_bm.png)




## 10. **동작 인식 및 비디오 분석 (Motion Analysis)**  
- **실습: 배경 차분법을 사용한 움직임 감지(OpenCV)**  
    - [샘플 동영상](https://pixabay.com/ko/videos/%EC%82%AC%EB%9E%8C%EB%93%A4-%EA%B1%B0%EB%A6%AC-%EC%9A%B0%ED%81%AC%EB%9D%BC%EC%9D%B4%EB%82%98-39836/)

```python
import cv2

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    cv2.imshow('Motion Detection', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

- **결과 예시**:

<video controls width="640" height="360">
    <source src="{{ site.url }}{{ site.baseurl }}/assets/videos/output_motion.mp4" type="video/mp4">   
    Your browser does not support the video tag.     
</video>

## 11. **평가 지표: 모델 성능 평가**
- 모델의 성능을 정량적으로 평가하기 위해 다양한 지표가 사용됩니다.

- **IoU(Intersection over Union)**
    - IoU는 객체 탐지 및 세그멘테이션에서 사용되는 지표로, 예측된 경계 상자와 실제 경계 상자의 겹치는 정도를 측정합니다.

    ```python
    import numpy as np

    def calculate_iou(boxA, boxB):
        """
        두 경계 상자(boxA, boxB)의 IoU를 계산합니다.
        boxA, boxB: [x_min, y_min, x_max, y_max]
        """
        # 겹치는 영역의 좌표 계산
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # 겹치는 영역의 넓이 계산
        inter_width = max(0, xB - xA + 1)
        inter_height = max(0, yB - yA + 1)
        inter_area = inter_width * inter_height

        # 각 박스의 넓이 계산
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # IoU 계산
        iou = inter_area / float(boxA_area + boxB_area - inter_area)
        return iou

    # 예시 경계 상자
    boxA = [50, 50, 150, 150]  # 실제 경계 상자
    boxB = [70, 60, 170, 140]  # 예측 경계 상자

    iou_score = calculate_iou(boxA, boxB)
    print(f"IoU: {iou_score:.2f}")
    ```


- **mAP(mean Average Precision)**
    - mAP는 객체 탐지에서 사용되며, 모델이 얼마나 정확하게 객체를 탐지했는지를 평가합니다.

    ```python
    from sklearn.metrics import average_precision_score

    # 예측 확률과 실제 레이블
    y_true = [1, 0, 1, 1, 0, 1]  # 실제 클래스 (1: Positive, 0: Negative)
    y_scores = [0.9, 0.4, 0.8, 0.7, 0.2, 0.6]  # 예측 확률

    # Average Precision 계산
    ap = average_precision_score(y_true, y_scores)
    print(f"Average Precision: {ap:.2f}")

    # mAP는 여러 클래스에 대해 AP를 평균한 값입니다.
    # 단일 클래스의 경우 AP와 동일합니다.
    ```

- **F1-score**
    - F1-score는 정밀도(Precision)와 재현율(Recall)의 조화 평균으로, 불균형한 데이터셋에서 유용

    ```python
    from sklearn.metrics import f1_score

    # 예측값과 실제값
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 0]

    # F1-score 계산
    f1 = f1_score(y_true, y_pred)
    print(f"F1-score: {f1:.2f}")
    ```

- **Confusion Matrix**

    ```python
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    # 예측값과 실제값
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 0]

    # Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 분류 보고서 출력
    print(classification_report(y_true, y_pred))
    ```


## 12. **응용 및 최신 트렌드**  
- **실습: GAN을 사용한 이미지 생성(DCGAN, PyTorch)**  

```python
import torch
from torchvision.utils import make_grid
from dcgan import Generator  # 사전 정의된 DCGAN Generator 모델

# Generator 모델 로드
netG = Generator().eval()
netG.load_state_dict(torch.load('generator.pth'))

# 랜덤 노이즈 생성 및 이미지 생성
noise = torch.randn(16, 100, 1, 1)
fake_images = netG(noise)

# 결과 시각화
grid = make_grid(fake_images, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0).detach().numpy())
plt.show()
```

![fake_image](/assets/images/fake_image.png)