---
title: 24차시 2:IBM TECH(Kubernetes Essentials)
layout: single
classes: wide
categories:
  - Kubernetes Essentials
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 11. Kubernetes Pod 생성
- 출처: [How does Kubernetes create a Pod?](https://www.youtube.com/watch?v=BgrQ16r84pM&list=PLOspHqNVtKABAVX4azqPIu6UfsPzSu2YN&index=12)


### 11.1 기본 구성 요소
* **노드 (Nodes):** Kubernetes 클러스터를 구성하는 작업 머신으로, 물리적 서버나 가상 머신이 될 수 있으며 워크로드를 실행하는 인프라의 기본 단위입니다.
* **컨트롤 노드 (Control Nodes):** 클러스터 관리 역할을 수행하는 특수 노드로, 워크로드 스케줄링, 상태 모니터링, API 처리 등 클러스터의 두뇌 역할을 합니다. (production 환경에서는 고가용성을 위해 3개 이상 권장)
* **컴퓨트 노드 (Compute Nodes):** 실제 워크로드(Pod, 컨테이너)가 실행되는 노드로, 리소스 요구사항에 따라 다양한 사양으로 구성될 수 있습니다. (워크로드 규모에 따라 수십~수백 개 존재 가능)
* **Kube-API 서버 (Kube-API Server):** Kubernetes 클러스터의 주요 관리 컴포넌트로, 모든 API 요청을 처리하고 다른 컴포넌트 간의 통신을 중개하는 중앙 집중식 허브 역할을 합니다.
* **etcd:** 클러스터의 모든 상태와 구성 정보를 저장하는 고가용성 분산 키-값 데이터 저장소로, 시스템의 '단일 진실 원천(source of truth)'으로 작동합니다.
* **스케줄러 (Scheduler):** 생성해야 할 워크로드를 감지하고, 노드의 리소스 상태, 제약조건, 정책 등을 고려하여 최적의 노드를 선택하는 지능적 배치 엔진입니다.
* **Kubelet:** 각 컴퓨트 노드에서 실행되는 에이전트로, 컨트롤 플레인(Kube-API 서버)과 통신하며 노드에서 컨테이너 실행을 관리합니다.
* **컨테이너 런타임 엔진 (Container Runtime Engine):** 컨테이너를 실제로 실행하는 소프트웨어로, Docker, containerd, CRI-O 등이 해당됩니다.
* **Kube 프록시 (Kube Proxy):** 노드 간 통신을 지원하는 네트워크 프록시로, 서비스 추상화를 구현하고 Pod 간의 네트워크 연결을 가능하게 합니다.
* **컨트롤러 매니저 (Controller Manager):** 클러스터의 상태를 지속적으로 감시하고, 원하는 상태와 실제 상태 간의 차이를 조정하는 다양한 컨트롤러를 관리합니다.

### 11.2 Pod 생성 과정
1. **Pod 생성 요청:** 
   * 사용자가 `kubectl` 명령어나 Kubernetes API를 사용하여 Pod 생성 요청을 보냅니다.
   * YAML이나 JSON 형식의 매니페스트 파일을 통해 Pod의 사양과 설정을 정의합니다.

2. **Kube-API 서버:**
   * 요청을 인증(Authentication)하여 사용자 신원을 확인하고, 권한 부여(Authorization)를 통해 해당 작업의 권한을 검증합니다.
   * Pod 정의에 대한 유효성 검사(Schema Validation)를 수행하여 필수 필드와 형식을 확인합니다.
   * 검증된 Pod 정보를 etcd에 '생성 중' 상태로 기록합니다.
   * etcd에 기록 성공 시 사용자에게 Pod 생성 요청 접수 응답을 보냅니다. (이 시점에는 실제 Pod 생성은 아직 완료되지 않음)

3. **스케줄러:**
   * Kube-API 서버의 이벤트를 감시하며 노드가 할당되지 않은 새로운 Pod를 감지합니다.
   * 클러스터 내 모든 컴퓨트 노드의 상태를 분석하고 다음 요소를 고려하여 Pod 배치 결정:
     * 노드의 하드웨어 리소스(CPU, 메모리) 가용성
     * 네트워크 요구사항 및 토폴로지
     * Pod의 노드 선호도/반선호도(affinity/anti-affinity) 설정
     * 테인트(taints)와 톨러레이션(tolerations) 설정
     * 사용자 정의 스케줄링 제약 조건
   * 최적의 노드를 선택한 후, Kube-API 서버에 Pod와 선택된 노드 간의 바인딩 정보를 업데이트합니다.

4. **Kube-API 서버 (다시):**
   * 스케줄러가 선택한 노드 정보를 Pod 정의에 업데이트하고 etcd에 기록합니다.
   * etcd 기록 성공 후, 해당 노드의 Kubelet에게 Pod 생성을 지시하는 이벤트를 발생시킵니다.

5. **Kubelet:**
   * 노드에 할당된 Pod에 대한 정보를 Kube-API 서버로부터 수신합니다.
   * Pod 정의에 명시된 볼륨을 생성하고 마운트 포인트를 준비합니다.
   * 필요한 시크릿(Secrets)과 컨피그맵(ConfigMaps)을 가져옵니다.
   * 컨테이너 런타임 엔진(CRI)에 컨테이너 이미지 다운로드 및 실행을 지시합니다.
   * 컨테이너 네트워크 인터페이스(CNI) 플러그인을 호출하여 Pod에 네트워크를 설정합니다.
   * Pod의 생명주기 이벤트(lifecycle events)와 상태를 모니터링하고 Kube-API 서버에 보고합니다.

6. **컨트롤러 매니저:**
   * 다양한 컨트롤러가 Pod의 상태를 지속적으로 감시합니다:
     * ReplicaSet 컨트롤러: 지정된 복제본 수를 유지
     * Deployment 컨트롤러: 롤링 업데이트 및 롤백 관리
     * StatefulSet 컨트롤러: 상태 유지가 필요한 애플리케이션 관리
     * DaemonSet 컨트롤러: 모든 노드에 Pod 실행 보장
   * Pod가 비정상 종료되거나 실패할 경우, 재시작 정책("Always", "OnFailure", "Never")에 따라 적절한 조치를 취합니다.
   * 필요시 새로운 Pod를 생성하여 원하는 상태를 유지합니다.

### 11.3 핵심
* Kubernetes는 선언적 모델(Declarative Model)을 사용합니다. 사용자가 원하는 상태(desired state)를 정의하면, Kubernetes 컴포넌트들이 협력하여 실제 상태(actual state)를 원하는 상태로 만들고 유지
* etcd는 클러스터의 모든 상태를 저장하는 핵심 컴포넌트로, 분산 시스템의 일관성과 내구성을 보장합니다.
* 각 컴포넌트들은 느슨하게 결합된 마이크로서비스 아키텍처로 구성되어 있으며, Kube-API 서버를 통해 비동기적으로 통신하고 etcd를 통해 상태를 공유합니다.
* 이러한 분산 아키텍처는 클러스터의 확장성과 복원력을 높이며, 개별 컴포넌트 실패에도 전체 시스템이 계속 작동할 수 있게 합니다.

## 12. Knative 란 무엇입니까?
- 출처: [Knative 란 무엇입니까?](https://www.youtube.com/watch?v=69OfdJ5BIzs&list=PLOspHqNVtKABAVX4azqPIu6UfsPzSu2YN&index=13)


### 12.1 Knative란?

* IBM, Google 등 업계 주요 기업들이 협력하여 개발한 Kubernetes 기반 플랫폼입니다. 이는 클라우드 컴퓨팅의 리더들이 모여 만든 오픈소스 프로젝트로, 서버리스 아키텍처의 장점을 Kubernetes 환경에 통합하고자 했습니다.
* Kubernetes 위에서 서버리스 워크로드를 실행할 수 있게 해주는 솔루션으로, 인프라 관리 부담을 줄이면서 개발자가 코드에 더 집중할 수 있게 합니다.
* 클라우드 네이티브 애플리케이션을 Kubernetes에서 더욱 효율적으로 빌드, 배포, 관리할 수 있도록 다양한 기능과 유틸리티를 제공합니다. 이를 통해 개발과 운영 사이의 간극을 좁힙니다.

### 12.2 Knative의 주요 구성 요소 (Primitives)

1. **Build:**
* 애플리케이션을 Kubernetes에 배포하는 복잡한 과정을 간소화하고 자동화합니다. 개발자는 배포 과정보다 코드 작성에 집중할 수 있습니다.
* 소스 코드 저장소 관리, 복잡한 빌드 프로세스 처리, Cloud Foundry 빌드 팩 템플릿 활용 등을 클러스터 내에서 원활하게 수행할 수 있습니다.
* 단일 매니페스트 배포 방식을 통해 개발 속도를 크게 향상시키고, 빠르게 변화하는 비즈니스 요구사항에 대응할 수 있는 민첩성을 확보합니다.

2. **Serve:**
* Istio 서비스 메시 컴포넌트가 내장되어 있어 트래픽 관리, 지능형 라우팅, 자동 스케일링, Scale to Zero(트래픽이 없을 때 리소스 사용량을 0으로 줄이는 기능) 등을 제공합니다.
* 서비스, Route, Configuration을 통해 애플리케이션을 체계적으로 관리할 수 있습니다.
* 코드를 푸시할 때마다 Revision(버전)을 자동으로 저장하여 버전 관리가 용이하고 롤백이 간편합니다.
* Istio의 강력한 트래픽 관리 기능을 활용하여 단계적 배포(staged rollout)나 A/B 테스트와 같은 고급 배포 전략을 쉽게 구현할 수 있습니다.
* 애플리케이션 상태의 스냅샷, 정교한 지능형 라우팅, 요구에 따른 자동 스케일링 기능을 제공합니다.
* CI/CD 파이프라인과 Kubernetes 마이크로서비스 배포 과정에서 발생하는 다양한 문제를 해결하는 데 도움을 줍니다.

3. **Eventing:**
* 서버리스 플랫폼의 핵심적인 기능인 트리거 생성과 이벤트 응답 메커니즘을 제공합니다. 이를 통해 이벤트 기반 아키텍처를 쉽게 구현할 수 있습니다.
* 다양한 이벤트 기반 동작을 설정할 수 있습니다. 예를 들어, 날씨 변화가 감지되면 배송 경로를 자동으로 재설정하는 알고리즘을 트리거하는 등의 복잡한 시나리오를 구현할 수 있습니다.
* CI/CD 파이프라인과 원활하게 연동되어, 마스터 브랜치에 코드가 푸시될 때 자동으로 특정 작업을 실행하도록 설정할 수 있습니다. 예를 들어, 트래픽의 10%를 새로운 버전의 애플리케이션으로 점진적으로 이동시키는 등의 고급 배포 전략을 자동화할 수 있습니다.

### 12.2 결론
* Knative는 클라우드 네이티브 및 Kubernetes 환경에서 개발과 운영을 간소화하고 자동화하는 강력한 도구입니다. 서버리스의 편의성과 Kubernetes의 강력함을 결합했습니다.
* Build, Serve, Eventing 세 가지 핵심 컴포넌트가 유기적으로 결합되어 시너지 효과를 창출함으로써, 개발자 경험을 향상시키고 운영 효율성을 극대화합니다.

## 13. Kubernetes 배포
- 출처: [Kubernetes 배포 : 빠른 시작](https://www.youtube.com/watch?v=Sulw5ndbE88&list=PLOspHqNVtKABAVX4azqPIu6UfsPzSu2YN&index=14)


### **13.1 Pod란?**
* Kubernetes 클러스터에서 실행되는 컨테이너의 가장 작은 단위입니다.
* 하나 이상의 컨테이너를 포함하며, 동일한 네트워크와 스토리지 자원을 공유합니다.
* Pod는 일시적인 존재이며, 장애 발생 시 Kubernetes에 의해 자동으로 재시작됩니다.

### **13.2 Kubernetes 배포(Deployment) 리소스**
* Pod를 선언적으로 관리하는 Kubernetes 리소스입니다.
* YAML 파일을 통해 원하는 상태(desired state)를 정의하고, Kubernetes는 이를 유지합니다.
* 롤링 업데이트, 롤백, 스케일링 등 다양한 기능을 제공하여 애플리케이션 배포를 간소화합니다.

### **13.3 YAML 파일 구조**
* **kind**: 리소스 종류를 정의합니다. (Deployment, Service, Ingress 등)
* **metadata**: 리소스의 이름, 네임스페이스, 레이블 등 메타 정보를 정의합니다.
* **spec**: 리소스의 원하는 상태를 정의합니다.
    * **replicas**: Pod 복제본 수를 지정합니다.
    * **selector**: Deployment가 관리할 Pod를 선택하는 레이블 셀렉터입니다.
    * **template**: Pod의 상세 정의를 포함합니다.
        * **containers**: Pod 내 컨테이너 목록과 이미지를 정의합니다.
        * **ports**: 컨테이너의 포트 정보를 정의합니다.
        * **env**: 컨테이너의 환경 변수를 정의합니다.
        * **volumes**: Pod에 연결할 스토리지 볼륨을 정의합니다.

### **13.4 배포 과정**
1.  `kubectl apply -f deployment.yaml` 명령어를 사용하여 YAML 파일을 클러스터에 배포합니다.
2.  API 서버는 Deployment 리소스를 생성하고, 컨트롤러 매니저에게 이를 전달합니다.
3.  Deployment 컨트롤러는 ReplicaSet을 생성하여 Pod 복제본 수를 관리합니다.
4.  ReplicaSet 컨트롤러는 Pod를 생성하고, 스케줄러는 Pod를 실행할 노드를 결정합니다.
5.  kubelet은 Pod를 노드에서 실행하고, 컨테이너 런타임은 컨테이너를 실행합니다.

### **13.5 업데이트 방법**
1.  YAML 파일을 수정하고 `kubectl apply -f deployment.yaml` 명령어를 실행합니다.
2.  Deployment 컨트롤러는 변경 사항을 감지하고 새로운 ReplicaSet을 생성합니다.
3.  기존 ReplicaSet의 Pod를 새로운 ReplicaSet의 Pod로 점진적으로 교체하는 롤링 업데이트를 수행
4.  롤링 업데이트 과정에서 서비스 중단 없이 애플리케이션을 업데이트할 수 있습니다.
5.  `kubectl rollout history deployment/<deployment-name>` 명령어를 사용하여 업데이트 이력을 확인하고, `kubectl rollout undo deployment/<deployment-name>` 명령어를 사용하여 이전 버전으로 롤백할 수 있습니다.

### **13.6 삭제 방법**
* `kubectl delete -f deployment.yaml` 명령어를 사용하여 Deployment 리소스를 삭제합니다.
* Deployment 컨트롤러는 ReplicaSet과 Pod를 삭제합니다.
* `kubectl delete deployment <deployment-name>` 명령어를 사용하여 이름을 통해 삭제할 수도 

### **13.7 디버깅 기법**
1.  **kubectl logs**: Pod 내 컨테이너의 로그를 확인하여 애플리케이션 오류를 진단합니다.
    * `--previous` 플래그를 사용하여 이전 컨테이너의 로그를 확인합니다.
2.  **kubectl describe pod**: Pod의 이벤트, 상태, 리소스 사용량 등 상세 정보를 확인합니다.
    * 잘못된 이미지, 포트 충돌, 리소스 부족 등 문제점을 파악합니다.
3.  **kubectl exec**: Pod 내 컨테이너에 명령어를 실행하여 디버깅합니다.
    * `kubectl exec -it <pod-name> -- /bin/bash` 명령어를 사용하여 Pod에 접속하고, `ps aux`, `netstat`, `curl` 등 다양한 명령어를 실행합니다.
4.  **kubectl port-forward**: 로컬 PC에서 Pod의 포트로 접근하여 애플리케이션을 테스트합니다.
5.  **kubectl events**: 클러스터에서 발생한 이벤트들을 확인하여 문제점을 파악합니다.
6.  **SSH를 통한 Pod 접속**:
    * 직접 Pod에 접속하여 프로세스 확인 (`ps aux`)
    * 파일 시스템에서 로그 확인

## 14. IBM CloudLabs
- 출처: [Using IBM CloudLabs for Hands-on Kubernetes Training on IBM Cloud](https://www.youtube.com/watch?v=6h-UCGJ4-BA&list=PLOspHqNVtKABAVX4azqPIu6UfsPzSu2YN&index=15)

### **14.1 IBM CloudLabs 란?**

*   **브라우저 기반으로 클라우드 네이티브 개념을 학습하는 인터랙티브 학습 플랫폼 (데모 환경 X)**  
    - IBM CloudLabs는 별도의 소프트웨어 설치 없이 웹 브라우저만으로 클라우드 네이티브 기술을 배울 수 있는 실습 중심의 환경을 제공합니다. 
    - 단순한 데모나 시연이 아니라, 사용자가 직접 명령어를 입력하고 결과를 확인하며 학습할 수 있는 플랫폼입니다.
*   **경험 수준에 관계없이 Kubernetes를 쉽게 시작 가능**  
    - 초보자부터 숙련자까지, Kubernetes와 같은 복잡한 클라우드 기술을 자신의 속도에 맞춰 학습할 수 있도록 설계되었습니다. 
    - 사전 지식이 없어도 단계별 안내를 통해 쉽게 접근할 수 있습니다.

### **14.2 사용 방법**

1.  **관심있는 랩 선택 후 내용 확인**  
    - 사용자는 제공된 다양한 주제의 랩 중에서 자신의 관심사나 학습 목표에 맞는 것을 선택하고, 해당 랩의 개요와 목표를 미리 확인할 수 있습니다.
2.  **우측 상단의 "Launch Lab" 버튼 클릭**  
    - 랩 설명 페이지에서 "Launch Lab" 버튼을 누르면 실습 환경으로 바로 이동할 준비가 됩니다.
3.  **로그인 (IBM Cloud 계정 필요, 무료)**  
    - 실습을 시작하려면 IBM Cloud 계정으로 로그인해야 하며, 계정이 없는 경우 무료로 가입 가능.
4.  **제공 시간 동안 랩 진행 (필요시 1시간 연장 가능)**  
    - 각 랩은 기본적으로 정해진 시간 동안 진행되며, 시간이 부족할 경우 1시간 추가 연장을 요청할 수 있어 유연한 학습이 가능합니다.
5.  **브라우저 내 터미널에서 안내에 따라 명령어 입력**  
    - 별도의 프로그램 설치 없이 브라우저에 내장된 터미널을 통해 제공된 가이드에 따라 명령어를 입력하고 실시간으로 결과를 확인하며 학습합니다.

### **14.3 장점**
*   **별도의 사전 설정이나 도구 설치 불필요**  
    - 복잡한 환경 설정이나 추가 소프트웨어 설치 없이, 인터넷 연결과 브라우저만 있으면 언제 어디서나 바로 실습을 시작할 수 있어 편리합니다.

## 15. Managed Kubernetes
- 출처: [관리되는 Kubernetes의 장점](https://www.youtube.com/watch?v=1Br4m0_8YDQ&list=PLOspHqNVtKABAVX4azqPIu6UfsPzSu2YN&index=16)


### 15.1 Kubernetes & Managed Kubernetes  

*   **Kubernetes:**  
    - Kubernetes는 컨테이너화된 애플리케이션을 자동으로 배포, 스케일링 및 운영하는 오픈 소스 플랫폼
    - 기존의 단일 서버 또는 가상머신(VM) 환경에서 애플리케이션을 실행하는 것과 달리, 컨테이너 기반 환경에서는 애플리케이션이 독립적인 컨테이너 단위로 실행되며, Kubernetes는 이러한 컨테이너들을 효과적으로 관리하고 조율하는 역할을 합니다.  

*   **Managed Kubernetes:**  
    클라우드 제공업체(AWS, Google Cloud, Azure, IBM Cloud 등)가 제공하는 관리형 Kubernetes 서비스로, 사용자가 직접 Kubernetes 클러스터를 구축하고 유지보수하는 부담을 덜어줍니다.
    *   **쉬운 클러스터 생성 및 확장:**  
        - 필요에 따라 즉시 새로운 Kubernetes 클러스터를 생성할 수 있으며, 클러스터의 컴퓨팅 리소스를 동적으로 조정할 수 있습니다. 이를 통해 수요 변화에 따라 원활하게 확장이 가능합니다.  
    *   **최신 오픈 소스 기술 및 클라우드 서비스 통합:**  
        - Managed Kubernetes 서비스는 Kubernetes의 최신 버전과 다양한 오픈 소스 기술을 쉽게 활용할 수 있도록 지원합니다. 
        - 또한, 클라우드 제공업체의 네트워크, 보안, 데이터 저장소 등의 서비스와 자연스럽게 통합되어 운영 효율성을 높일 수 있습니다.  
    *   **강력한 보안:**  
        - 관리형 Kubernetes 서비스는 기본적으로 강화된 보안 설정을 제공하며, 기업이 요구하는 보안 표준을 충족할 수 있도록 도와줍니다. 
        - 자동 패치, 암호화된 통신, 네트워크 정책 적용 등을 통해 안정적인 환경을 제공합니다.  

### 15.2 IBM Cloud Kubernetes Service를 통한 클러스터 생성 예시  

Managed Kubernetes의 대표적인 서비스 중 하나로, IBM Cloud Kubernetes Service를 사용하면 손쉽게 클러스터를 생성하고 관리할 수 있습니다. 클러스터 생성 과정은 다음과 같습니다.  

*   **클러스터 이름 및 지역 선택**  
    - 사용자는 클러스터를 배포할 데이터 센터의 위치를 지정할 수 있습니다. 
    - 단일 영역 클러스터뿐만 아니라, 여러 데이터 센터에 걸쳐 배포되는 다중 영역 클러스터(Multi-Zone Cluster)를 선택하여 고가용성을 보장할 수도 있습니다.  

*   **컴퓨팅 파워 선택**  
    - Kubernetes 클러스터를 실행할 컴퓨팅 인스턴스를 선택할 수 있으며, 기본적으로 가상 컴퓨팅(Virtual Machines)과 물리 서버(Bare Metal)를 지원합니다.  
        *   **Bare Metal:** "Noisy Neighbor" 문제 없이 단독으로 머신을 사용할 수 있어, 성능이 중요한 워크로드에 적합합니다.  
        *   **GPU 옵션:** 머신러닝(ML), 딥러닝(DL), 이미지/비디오 처리 등 GPU가 요구되는 고성능 연산 작업에 활용 가능합니다.  

*   **워크 노드 수 선택 후 클러스터 생성**  
    - 워커 노드(Worker Nodes)의 개수를 지정한 후, 클러스터를 생성하면 몇 분 안에 프로비저닝이 완료
    - 생성된 클러스터는 IBM Cloud Kubernetes Service의 관리 도구를 통해 쉽게 모니터링 및 운영

### 15.3 오픈 소스 및 표준 준수  

Managed Kubernetes 환경에서는 표준을 준수하는 것이 중요한데, IBM Cloud Kubernetes Service는 오픈 소스 표준을 기반으로 구축되어 있어 여러 클라우드 환경에서의 이동성과 확장성을 제공합니다.  

*   **CNCF (Cloud Native Computing Foundation):**  
    - Kubernetes는 CNCF의 대표적인 프로젝트 중 하나로, CNCF는 클라우드 네이티브 애플리케이션을 위한 다양한 오픈 소스 기술을 지원합니다.  

*   **적합성 테스트:**  
    - CNCF는 Kubernetes 서비스 제공업체 간의 호환성을 유지하기 위해 적합성 테스트를 수행하며, 
    - 이를 통해 사용자는 특정 클라우드 제공업체에 종속되지 않고 여러 환경에서 동일한 Kubernetes 워크로드를 실행할 수 있습니다.  

*   **IBM의 오픈 소스 기여:**  
    - IBM은 Kubernetes를 비롯해 Istio(서비스 메쉬), Knative(서버리스 컴퓨팅) 등의 오픈 소스 프로젝트를 적극 지원하며, 
    - IBM Cloud Kubernetes Service 내에서 Managed Istio 및 Managed Knative를 기본적으로 제공하여 사용자가 손쉽게 활용할 수 있도록 합니다.  
    - 또한, IBM은 다양한 튜토리얼과 문서를 통해 최신 오픈 소스 기술과 IBM 도구/서비스를 쉽게 통합할 수 있도록 지원합니다.  

### 15.4 보안  

Managed Kubernetes 환경에서는 보안이 중요한 요소이며, IBM Cloud Kubernetes Service는 DevOps 워크플로우의 모든 단계에서 강력한 보안을 제공합니다.  

*   **신뢰할 수 있는 프라이빗 레지스트리:**  
    - IBM Cloud Container Registry를 통해 컨테이너 이미지를 안전하게 저장하고 관리할 수 있으며, 이미지의 취약점을 사전에 검사하여 보안 위협을 예방할 수 있습니다.  

*   **이미지 스캔:**  
    - Kubernetes 클러스터에 컨테이너 이미지를 배포하기 전에 자동으로 보안 취약점이 있는지 스캔하여 안전성을 확보합니다.  

*   **지속적인 취약점 관리:**  
    - Kubernetes 클러스터를 구성하는 여러 요소(컨테이너 런타임, 네트워크, 스토리지 등)의 보안 취약점을 지속적으로 모니터링하고 즉각적인 조치를 취할 수 있도록 지원합니다.  
    - 또한, 정책 기반 보안 관리를 통해 특정 보안 기준을 준수하도록 강제할 수 있습니다.  
