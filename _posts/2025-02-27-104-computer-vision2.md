---
title: 12차시 2:Computer Vision(Fine-Tuning)
layout: single
classes: wide
categories:
  - Computer Vision
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## **1.전이학습**

- 아래는 **CIFAR-10 데이터셋**을 활용하여 사전 학습된 ResNet 모델을 Fine-tuning하는 전체 코드입니다. 학습 후에는 테스트 데이터셋으로 모델의 성능을 평가하고, Confusion Matrix와 분류 보고서를 출력

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 2. 데이터셋 로딩 (CIFAR-10)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 10% 데이터 사용🚀
# train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), 10))  

# 10% 데이터 사용🚀
# test_dataset = torch.utils.data.Subset(test_dataset, indices=range(0, len(test_dataset), 10))  

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 사전 학습된 ResNet 모델 로드 및 수정
model = models.resnet18(pretrained=True)

# 마지막 레이어 수정 (CIFAR-10은 10개 클래스)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 출력 레이어를 10개 클래스로 변경

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()

# 모델 전체 학습🚀(Fine-Tuning)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 마지막 레이어만 학습🚀(Feature Extraction)
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  

# 5. 학습 루프
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("학습 완료!")

# 6. 평가 함수
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"테스트 정확도: {accuracy:.2f}%")

    # Confusion Matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # 분류 보고서 출력
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# 7. 학습 및 평가 실행
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
evaluate_model(model, test_loader)
```

1.  **데이터 전처리**
- `transforms.Resize((224, 224))`: ResNet은 224x224 크기의 이미지를 입력으로 받습니다.
- `transforms.Normalize`: ImageNet 데이터셋의 평균과 표준편차를 기반으로 정규화합니다.

2.  **데이터셋 로딩**
- CIFAR-10 데이터셋은 10개의 클래스로 구성된 이미지 데이터셋입니다.
- `train_loader`와 `test_loader`를 통해 학습 및 테스트 데이터를 배치 단위로 제공합니다.

3.  **사전 학습된 모델 수정**
- ResNet의 마지막 Fully Connected Layer(`fc`)를 CIFAR-10의 10개 클래스에 맞게 수정합니다.

4.  **학습 루프**
- `train_model` 함수는 주어진 에포크 동안 모델을 학습합니다.
- 각 배치마다 손실을 계산하고, 역전파를 통해 모델 파라미터를 업데이트합니다.

5.  **평가 함수**
- `evaluate_model` 함수는 테스트 데이터셋으로 모델을 평가하고, 정확도를 계산합니다.
- Confusion Matrix와 분류 보고서를 통해 모델의 성능을 시각적으로 확인할 수 있습니다.


🔹 **출력 예시:** 
- **테스트 정확도**
```
Epoch [1/5], Loss: 0.5625
Epoch [2/5], Loss: 0.5484
Epoch [3/5], Loss: 0.5357
Epoch [4/5], Loss: 0.5289
Epoch [5/5], Loss: 0.5363
학습 완료!
테스트 정확도: 75.50%
```

- **Confusion Matrix**
![Confusion Matrix](/assets/images/feature_extraction_cm.png)

- **분류 보고서**
```
           precision    recall  f1-score   support

    airplane       0.60      0.89      0.72        87
  automobile       0.89      0.90      0.90       100
        bird       0.78      0.63      0.70       108
         cat       0.63      0.64      0.64       107
        deer       0.68      0.73      0.70        95
         dog       0.62      0.72      0.66        95
        frog       0.81      0.77      0.79       100
       horse       0.83      0.68      0.75       102
        ship       0.88      0.79      0.84       102
       truck       0.94      0.84      0.88       104

    accuracy                           0.76      1000
   macro avg       0.77      0.76      0.76      1000
weighted avg       0.77      0.76      0.76      1000
```

