---
title: 15차시 5:n8n (Local host access from outside)
layout: single
classes: wide
categories:
  - n8n
  - ellevenlabs
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

# ngrok설치하여: 외부서비스에서 로컬호스트(localhost)에 접근하기

## 1. 설치 방법(다음 방법 중 하나를 선택)

1. 직접 다운로드 방식
    - [ngrok 공식 웹사이트](https://ngrok.com/download)에서 Windows용 ZIP 파일을 다운로드합니다.
    - 다운로드한 ZIP 파일을 원하는 위치에 압축 해제합니다.
    - 압축 해제된 폴더에 `ngrok.exe` 파일이 있습니다.

2. Chocolatey 패키지 관리자 사용 (선택사항)
    이미 Chocolatey가 설치되어 있다면 다음 명령어를 관리자 권한 명령 프롬프트에서 실행:
    ```
    choco install ngrok
    ```

3. Windows 패키지 관리자(winget) 사용 (Windows 10 이상)
    ```
    winget install ngrok.ngrok
    ```

## 2. 인증 설정
1. [ngrok.com](https://ngrok.com)에서 무료 계정을 만듭니다.
2. 대시보드에서 인증 토큰을 복사합니다.
3. 명령 프롬프트(CMD)나 PowerShell을 열고 다음 명령어 실행:
```
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

## 3. 실행 방법
1. 명령 프롬프트(CMD)나 PowerShell을 엽니다.
2. ngrok.exe가 있는 폴더로 이동하거나, Chocolatey/winget으로 설치했다면 어디서든 실행 가능합니다.
3. n8n을 위한 5678 포트를 외부에 노출하려면 다음 명령어 실행:
```
ngrok http 5678
```
4. 명령어가 실행되면 다음과 같은 화면이 나타나고 포워딩 URL이 표시됩니다:
   - `https://xxxx-xxx-xxx-xxx-xxx.ngrok-free.app` 형태의 URL을 메모해두세요
   - 이 URL을 ElevenLabs webhook URL로 사용하면 됩니다

## 4. 주의사항
- 무료 계정은 세션이 8시간 후 자동 종료되며, URL이 매번 변경됩니다.
- 명령 프롬프트 창을 닫으면 ngrok 연결도 종료됩니다.
- 고정 URL이 필요하면 유료 플랜으로 업그레이드해야 합니다.
