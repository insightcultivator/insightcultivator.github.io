---
title: 15차시 3:n8n (Load To PineCone)
layout: single
classes: wide
categories:
  - n8n
  - pinecone
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

- **구글 드라이브에 있는 파일들을 PineCone 벡터 데이터베이스로 저장하기**

![Load To Pine Cone](/assets/images/load_to_pinecone.png)

## 1. Credential 생성하기
- n8n 워크플로우에서 사용되는 각 노드는 특정 API 또는 서비스에 접근하기 위해 자격 증명(credentials)을 요구할 수 있습니다. 상기 프로젝트의 경우, 다음과 같은 credential이 필요할 가능성이 높습니다:

### 1.1 **Google Drive (search fileFolder 및 download file)**  
- **필요한 Credential**: Google OAuth 2.0 인증 정보  
  - Google Drive API를 사용하여 폴더 검색 및 파일 다운로드를 수행하려면 OAuth 2.0 기반의 인증이 필요합니다. 이는 Google Cloud Console에서 생성된 클라이언트 ID와 클라이언트 시크릿을 통해 이루어집니다.

- **구체적인 단계**:
  - Google Drive 관련 노드 생성시 구성할 것.
  - [Google Cloud Console](https://console.cloud.google.com/)에서 새 프로젝트 생성.
  - "API 및 서비스" > "라이브러리"에서 **Google Drive API** 활성화.
  - "OAuth 동의 화면" 구성.
  - "API 및 서비스" > "자격 증명"에서 OAuth 2.0 클라이언트 ID 생성.
  - 생성된 클라이언트 ID와 클라이언트 시크릿을 n8n의 Google Drive 노드에 입력.

### 1.2 **Pinecone Vector Store**  
- **필요한 Credential**: Pinecone API 키  
  - Pinecone은 벡터 데이터베이스로, 데이터를 저장하고 검색하기 위해 API 키가 필요합니다. 이 키는 Pinecone 계정에서 발급받을 수 있습니다.
- **구체적인 단계**:
  - [Pinecone 콘솔](https://www.pinecone.io/)에 로그인.
  - 프로젝트를 생성하고 API 키를 발급받음.
  - 발급받은 API 키를 n8n의 Pinecone 노드에 입력.

### 1.3 **Embeddings Google Gemini**  
- **필요한 Credential**: Google AI Studio API 키 또는 Google Cloud Service Account Key  
  - Google Gemini는 Google의 대형 언어 모델(LLM)로, 이를 사용하려면 Google AI Studio 에서 제공하는 API 키가 필요합니다.
- **구체적인 단계**:
  - [Google AI Studio](https://aistudio.google.com/)에 접속.
  - API 키 또는 서비스 계정 키(JSON 파일) 발급.
  - 발급받은 키를 n8n의 Google Gemini 노드에 입력.


### 1.4 요약: 필수 Credentials


| **노드**                     | **필요한 Credential**             | **발급 방법**                                                                 |
|-------------------------------|-----------------------------------|-------------------------------------------------------------------------------|
| Google Drive (검색 및 다운로드) | Google OAuth 2.0 클라이언트 ID/시크릿 | Google Cloud Console에서 생성                                                |
| Pinecone Vector Store         | Pinecone API 키                  | Pinecone 콘솔에서 발급                                                       |
| Embeddings Google Gemini      | Google AI Studio API 키 또는 GCP 서비스 계정 키 | Google AI Studio 또는 Google Cloud Console에서 발급                          |

## 2. 구글 드라이브에 있는 파일을 PineCone 벡터 스토어로 저장
### 2.1 **When clicking 'Test workflow'**
- 이 단계는 n8n 워크플로우 실행 트리거를 나타냅니다. 사용자가 '테스트 워크플로우(Test workflow)' 버튼을 클릭하면, 설정된 노드들 간의 연결 및 데이터 처리가 순차적으로 실행됩니다. 이는 전체 워크플로우를 디버깅하거나 실제 환경에서 작동 여부를 확인하기 위한 수동 트리거입니다.

### 2.2 **Google Drive: search fileFolder**
- Google Drive API를 활용하여 특정 폴더 또는 파일을 검색합니다. 이때 검색 조건(예: 폴더 이름, 파일 형식 등)을 지정하여 원하는 대상만 필터링할 수 있습니다. 이 단계에서는 Google Drive 내 폴더 ID나 파일 경로를 추출하여 다음 단계에서 사용할 수 있도록 준비합니다.


### 2.3 **Google Drive1: download file**
- 앞서 검색한 폴더에서 파일을 다운로드하는 과정입니다. 이때 HTTP 요청을 통해 Google Drive 서버로부터 파일 데이터를 받아오며, 로컬 시스템이나 워크플로우 내에서 처리 가능한 상태로 변환합니다.

### 2.4 **Loop Over Items**
- 다운로드한 여러 파일들을 하나씩 반복적으로 처리하기 위해 루프를 사용합니다. n8n의 `Loop Over Items` 노드는 배열 형태의 데이터를 반복적으로 순회하며, 각 항목에 대해 동일한 작업을 적용할 수 있도록 설계되었습니다. 이는 복수의 파일이나 데이터 집합을 처리해야 할 때 매우 유용합니다.

### 2.5 **Pinecone Vector Store**
- Pinecone은 고성능 벡터 저장소(Vector Database)로, 비정형 데이터(텍스트, 이미지 등)를 벡터로 변환하여 저장하고 검색할 수 있는 서비스입니다. 여기서는 파일의 내용을 벡터화한 후 Pinecone에 저장하여 이후 자연어 처리(NLP)나 의미론적 검색(Semantic Search)에 활용할 수 있도록 합니다.
- 차원수를 맞추어야 한다
  - Google Gemini Embeddings: 768차원 (일반적인 경우).
  - OpenAI Embeddings (text-embedding-ada-002): 1536차원 .
  - 다른 임베딩 모델: 각 모델마다 고유한 차원 수를 가짐.

### 2.6 **Embeddings Google Gemini**
- Google Gemini는 대형 언어 모델(LLM)로, 입력된 텍스트를 임베딩(Embedding)이라고 불리는 고차원 벡터 공간으로 변환합니다. 임베딩은 텍스트의 의미적 특성을 수치적으로 표현하며, 이를 통해 유사한 문장이나 개념을 계산적으로 비교할 수 있습니다.

### 2.7 **Default Data Loader**
- 데이터 로더(Data Loader)는 다양한 소스(파일, 데이터베이스 등)에서 데이터를 읽어오는 역할을 합니다. 기본 데이터 로더는 주로 텍스트 파일이나 CSV 등의 구조화된 데이터를 불러와 후속 처리 단계에 필요한 형태로 변환합니다.


### 2.8 **Recursive Character Text Splitter**
- 재귀적 문자 분할(Recursive Character Text Splitter)은 텍스트를 특정 규칙에 따라 계층적으로 분리하는 방법입니다. 예를 들어, 먼저 문단으로 나눈 후 문장을 나누고, 다시 문장을 단어로 나누는 방식으로 진행됩니다. 이는 텍스트의 구조를 유지하면서도 세밀하게 분석할 수 있도록 돕습니다.

### 2.10 요약
1. **트리거**를 통해 수동으로 시작.
2. **Google Drive**에서 파일 검색 및 다운로드.
3. **루프**를 통해 다운로드한 파일들을 하나씩 처리.
4. **Pinecone**과 **Google Gemini**를 활용하여 데이터를 벡터화하고 저장.
5. **데이터 로더**와 **텍스트 스플리터**를 통해 텍스트를 분석 가능한 형태로 가공.

## 3. 각 과정의 상세 설명
### 3.1 **When clicking 'Test workflow'**

### 3.2 **Google Drive: search fileFolder**
![Google Drive search FileFolder](/assets/images/gd_search.png)

### 3.3 **Google Drive1: download file**
![Google Drive download files](/assets/images/gd_download.png)

### 3.4 **Loop Over Items**
- Loop Over Items의 노드에서 Replace me를 삭제하고 그곳에 Pinecone Vector Store 연결

### 3.5 **Pinecone Vector Store**
![Pinecone Vector store](/assets/images/loadtopinecone.png)

### 3.6 **Embeddings Google Gemini**
![Embeddings Google Gemini](/assets/images/load_embedding_gemini.png)

### 3.7 **Default Data Loader**
![Default Data Loader](/assets/images/default_data_loader.png)

### 3.8 **Recursive Character Text Splitter**
![Text Splitter](/assets/images/text_splitter.png)

`