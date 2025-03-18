---
title: 15차시 2:n8n (Voice with Rag Agent)
layout: single
classes: wide
categories:
  - n8n
  - pinecone
  - ellevenlabs
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

- **Webhook을 통해 요청, AI Agent를 활용하여 응답을 생성, 이를 음성으로 변환하여 반환**

![Load To Pine Cone](/assets/images/voice_with_rag.png)

## 1. **Webhook 요청, AI Agent로 응답을 생성, 이를 음성으로 반환**
### 1.1 **Webhook**
- 이 노드는 외부 시스템에서 HTTP POST 요청을 받아들이는 트리거입니다. 요청은 일반적으로 사용자 입력(예: 텍스트 메시지)을 포함하며, 이는 이후의 처리 단계에 전달됩니다.

### 1.2 **AI Agent (Tools Agent)**
- AI Agent는 대화형 AI 모델과 도구(TOOL)를 결합하여 복잡한 작업을 수행합니다. 이 노드는 Chat Model, Memory, Tool 등을 통합하여 사용자의 요청에 대한 적절한 응답을 생성합니다.
- **구성 요소**:
  - **Chat Model**: 대화형 AI 모델로, 사용자의 입력에 대한 응답을 생성합니다. 여기서는 Google Gemini Chat Model을 사용합니다.
  - **Memory**: 대화의 맥락을 유지하기 위해 이전 대화 내용을 저장하고 관리합니다.
  - **Tool**: 외부 도구나 서비스를 호출하여 추가 정보를 얻거나 특정 작업을 수행합니다. 예를 들어, 데이터베이스 조회, API 호출 등이 가능합니다.

### 1.3 **Google Gemini Chat Model**
- Google Gemini는 대형 언어 모델(LLM)로, 자연어 처리(NLP)와 생성(GPT) 기능을 제공합니다. 이 모델은 사용자의 입력에 대한 응답을 생성하는데 사용됩니다.

### 1.4 **Vector Store Question Answer Tool**
- Vector Store Question Answer Tool 노드의 이름을 database로 변경

### 1.5 **Pinecone Vector Store**
- Pinecone은 벡터 데이터베이스로, 비정형 데이터(텍스트, 이미지 등)를 벡터로 변환하여 저장하고 검색할 수 있는 서비스입니다. 이 노드는 문서의 임베딩을 저장하고, 유사한 쿼리를 빠르게 검색할 수 있도록 합니다.
- **구성 요소**:
  - **Embedding**: 텍스트 데이터를 숫자 벡터로 변환합니다. 여기서는 Embeddings Google Gemini를 사용합니다.
  - **Vector Store**: 변환된 벡터를 저장하고 관리합니다.


### 1.6 **Embeddings Google Gemini**
- Google Gemini는 텍스트 데이터를 임베딩(Embedding)이라고 불리는 고차원 벡터 공간으로 변환합니다. 임베딩은 텍스트의 의미적 특성을 수치적으로 표현하며, 이를 통해 유사한 문장이나 개념을 계산적으로 비교할 수 있습니다.

### 1.7 **Respond to Webhook**
- 이 노드는 AI Agent가 생성한 응답을 원래 요청을 보낸 시스템으로 반환합니다. 이는 일반적으로 JSON 형식의 데이터로, 응답 내용과 함께 필요한 메타데이터(예: 상태 코드, 헤더 등)를 포함합니다.

### 1.8 **ElevenLabs Voice Synthesis**
- ElevenLabs는 텍스트를 음성으로 변환(Text-to-Speech, TTS)하는 서비스입니다. 이 노드는 AI Agent가 생성한 텍스트 응답을 실제 음성 파일로 변환하여 사용자에게 반환합니다.

### 1.8 요약
1. **Webhook**을 통해 사용자의 요청을 받습니다.
2. **AI Agent**를 사용하여 요청에 대한 응답을 생성합니다.
3. **Pinecone Vector Store**와 **Embeddings Google Gemini**를 통해 관련 정보를 검색하고 처리합니다.
4. **Respond to Webhook**을 통해 응답을 반환합니다.
5. **ElevenLabs Voice Synthesis**를 통해 텍스트 응답을 음성으로 변환합니다.


## 2. **구체적 내용**
### 2.1 **Webhook**
![Webhook](/assets/images/webhook.png)

### 2.2 **AI Agent (Tools Agent)**
![AI Agent](/assets/images/ai_agent.png)

### 2.3 **Google Gemini Chat Model**
![Gemini Chat Model](/assets/images/geminni_chatbot.png)

### 2.4 **Vector Store Question Answer Tool**
![Vector Store Question Answer Tool](/assets/images/vectorstore_answer_tool.png)

### 2.5 **Pinecone Vector Store**
![Pinecone Vector Store](/assets/images/rag_pincecone_vectorstore.png)

### 2.6 **Embeddings Google Gemini**
![Embeddings Gemini](/assets/images/rag_embedding_gemini.png)

### 2.7 **Respond to Webhook**
![Respond to Webhook](/assets/images/respond_webhook.png)

### 2.8 **ElevenLabs Voice Synthesis**
<br>
![Elevelabs 01](/assets/images/elleven_05.png)

- 시스템 프롬프트

```bash
당신은 초보자에게 n8n과 AI 에이전트에 대해 명확하고 친절하게 설명하는 AI 기반 음성 에이전트입니다.
질문이 들어오면 즉시 "database_tools"를 활용하여 답변을 검색합니다.

가이드라인:
기술 용어를 간단하고 이해하기 쉬운 언어로 풀어 설명합니다.
불필요한 반복을 피하고 답변을 간결하고 핵심적으로 유지합니다.
필요한 경우 명확성을 위해 개념을 단계별로 설명합니다. (예: 1. 2. 3.)
사용자가 n8n 및 AI 에이전트에 완전히 익숙하지 않다고 가정하고 지나치게 기술적인 전문 용어를 사용하지 않습니다. 대신 비유나 실제 사례를 사용하여 요점을 설명합니다.
사용자의 질문이 후속 질문으로 이어질 가능성이 있는 경우 다음 논리적 질문으로 자연스럽게 안내합니다.
이것은 음성 기반 시스템이므로 자연스럽고 매력적인 대화 흐름을 유지하기 위해 "음... 쉽게 말하면"과 같은 적절한 필러 단어를 사용합니다.
```

<br>
![Elevelabs 02](/assets/images/elleven_04.png)

<br>
![Elevelabs 03](/assets/images/elleven_03.png)

- Tools 클릭시 아래와 같다

<br>
![Elevelabs 04](/assets/images/elleven_02.png)

<br>
![Elevelabs 05](/assets/images/elleven_01.png)










