---
title: 14차시 2:FLASK 예시
layout: single
classes: wide
categories:
  - FLASK
toc: true # 이 포스트에서 목차를 활성화
toc_sticky: true # 목차를 고정할지 여부 (선택 사항)
---

## 1. 샘플 프로젝트
### 1.1 프로젝트 구조 (업데이트)
```
flask_app/
│
├── app.py                # Flask 애플리케이션의 메인 파일
├── templates/
│   ├── login.html        # 로그인 페이지
│   ├── register.html     # 회원가입 페이지
│   └── main.html         # 메인 페이지
└── static/
    └── styles.css        # CSS 스타일 파일
```


### 1.2 `app.py` (Flask 애플리케이션 업데이트)
```python
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 세션 암호화를 위한 시크릿 키


# 데이터베이스 초기화 함수
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# 데이터베이스에서 사용자 조회
def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


# 데이터베이스에 사용자 추가
def add_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("main"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if not username or not password:
            flash("모든 필드를 입력해주세요.", "danger")
            return redirect(url_for("register"))

        if add_user(username, password):
            flash("회원가입이 완료되었습니다. \n로그인해주세요.", "success")
            return redirect(url_for("login"))
        else:
            flash("이미 존재하는 아이디입니다.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = get_user(username)
        if user and user[2] == password:  # user[2]는 비밀번호
            session["username"] = username
            flash("로그인에 성공했습니다!", "success")
            return redirect(url_for("main"))
        else:
            flash("아이디 또는 비밀번호가 잘못되었습니다.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/main")
def main():
    if "username" not in session:
        flash("로그인이 필요합니다.", "warning")
        return redirect(url_for("login"))
    return render_template("main.html", username=session["username"])


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("로그아웃 되었습니다.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()  # 애플리케이션 시작 시 데이터베이스 초기화
    app.run(debug=True)

```


### 1.3 `templates/register.html` (회원가입 페이지)
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>회원가입</title>
    {% raw %}
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    {% endraw %}
</head>
<body>
    <div class="login-container">
        <h2>회원가입</h2>
        {% raw %}
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <ul class="flash-messages">
                    {% for category, message in messages %}
                    <li class="{{ category }}">{{ message|replace('\n', '<br>')|safe }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
        {% endraw %}
        {% raw %}<form method="POST" action="{{ url_for('register') }}">{% endraw %}
            <label for="username">아이디:</label>
            <input type="text" id="username" name="username" required>
            <label for="password">비밀번호:</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">회원가입</button>
        </form>
        {% raw %}<p>이미 계정이 있으신가요? <a href="{{ url_for('login') }}">로그인하기</a></p>{% endraw %}
    </div>
</body>
</html>
```


### 1.4 `templates/login.html` 
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>로그인</title>
   {% raw %} <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">{% endraw %}
</head>
<body>
    <div class="login-container">
        <h2>로그인</h2>
        {% raw %}
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <ul class="flash-messages">
                    {% for category, message in messages %}
                        <li class="{{ category }}">{{ message }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
        {% endraw %}
        {% raw %}<form method="POST" action="{{ url_for('login') }}">{% endraw %}
            <label for="username">아이디:</label>
            <input type="text" id="username" name="username" required>
            <label for="password">비밀번호:</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">로그인</button>
        </form>
        {% raw %}<p>계정이 없으신가요? <a href="{{ url_for('register') }}">회원가입하기</a></p>{% endraw %}
    </div>
</body>
</html>

```

### 1.5 `templates/main.html` 메인 페이지
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>메인 페이지</title>
    {% raw %}
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">{% endraw %}
</head>
<body>
    <div class="main-container">
        {% raw %}
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <ul class="flash-messages">
                    {% for category, message in messages %}
                        <li class="{{ category }}">{{ message }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
        {% endraw %}
        <h2>환영합니다, {{ username }}님!</h2>
        <p>이곳은 메인 페이지입니다.</p>
        {% raw %}<a href="{{ url_for('logout') }}" class="logout-button">로그아웃</a>{% endraw %}
    </div>
</body>
</html>
```

### 1.6 `static/styles.css` 
```html
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.login-container, .main-container {
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    width: 300px;
    text-align: center;
}

h2 {
    margin-bottom: 20px;
}

label {
    display: block;
    margin-top: 10px;
    font-weight: bold;
}

/* 로그인 및 회원가입 폼의 입력 필드 스타일 */
input[type="text"], input[type="password"] {
    width: 100%;
    padding: 12px; /* 패딩 추가 */
    margin-top: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* 내부 패딩과 테두리를 포함한 크기 계산 */
    font-size: 16px; /* 글꼴 크기 조정 */
}

button {
    margin-top: 20px;
    padding: 10px;
    width: 100%;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background-color: #0056b3;
}

.logout-button {
    display: inline-block;
    margin-top: 20px;
    padding: 10px 20px;
    background-color: #dc3545;
    color: white;
    text-decoration: none;
    border-radius: 4px;
}

.logout-button:hover {
    background-color: #c82333;
}

.flash-messages {
    list-style: none;
    padding: 0;
    margin-bottom: 15px;
}

.flash-messages li {
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 5px;
}

.success {
    background-color: #d4edda;
    color: #155724;
}

.danger {
    background-color: #f8d7da;
    color: #721c24;
}

.warning {
    background-color: #fff3cd;
    color: #856404;
}

.flash-messages li {
    white-space: pre-line;
  }
```

### 1.7 실행 방법
1. 위의 파일들을 적절한 디렉토리 구조로 작성합니다.
2. 터미널에서 `app.py`가 있는 디렉토리로 이동하여 다음 명령어를 실행합니다:
   ```bash
   python app.py
   ```
3. 브라우저에서 `http://127.0.0.1:5000/register`로 접속하여 회원가입 페이지를 확인합니다.
4. 회원가입 후 로그인하여 메인 페이지로 이동합니다.



### 1.8 주요 사항
1. **SQLite 데이터베이스**:
   - `users.db`라는 SQLite 데이터베이스를 사용하여 사용자 정보를 저장합니다.
   - `users` 테이블에는 `id`, `username`, `password` 컬럼이 있습니다.

2. **회원가입 기능**:
   - `/register` 경로를 추가하여 회원가입 페이지를 제공합니다.
   - 중복된 아이디를 방지하기 위해 데이터베이스 제약 조건(`UNIQUE`)을 사용합니다.

3. **데이터베이스 연동**:
   - `sqlite3` 모듈을 사용하여 데이터베이스와 상호작용합니다.

4. **플래시 메시지**:
   - 회원가입 및 로그인 과정에서 피드백을 제공합니다.


## 2. PythonAnywhere로 배포
### **2.1 PythonAnywhere 계정 생성**
- [PythonAnywhere](https://www.pythonanywhere.com/)에 접속하여 계정을 생성합니다.
- 무료 계정으로도 Flask 애플리케이션을 배포할 수 있지만, 제한된 리소스와 기능을 제공합니다.

### **2.2 로컬 환경에서 Flask 애플리케이션 준비**

- 필요한 파일:
    - `app.py`: 메인 Flask 애플리케이션 파일.
    - `requirements.txt`: 프로젝트의 의존성 목록을 포함한 파일.
    - `requirements.txt`는 다음과 같이 작성할 수 있습니다:
    ```
    Flask==2.3.2
    ```

### **2.3 Git 저장소에 코드 업로드**
PythonAnywhere에서는 Git을 통해 코드를 가져올 수 있습니다. 따라서 GitHub 또는 다른 Git 호스팅 서비스에 코드를 업로드해야 합니다.

1. Git 저장소 초기화:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. GitHub에 저장소 생성 후 코드 푸시:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```



### **2.4 PythonAnywhere Bash 콘솔에서 작업**
PythonAnywhere 대시보드에서 Bash 콘솔을 열고 다음 명령어를 실행합니다.

- Git 클론 혹은 파일 업로드
    - GitHub 저장소를 PythonAnywhere에 복제합니다:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

- 가상 환경 설정
    - PythonAnywhere에서 가상 환경을 생성하고 활성화합니다:
    ```bash
    python -m venv .venv --python=python3.9
    pip install -r requirements.txt
    ```

- WSGI 파일 구성
    - PythonAnywhere는 WSGI(Web Server Gateway Interface)를 사용하여 Flask를 실행.
        - PythonAnywhere 대시보드에서 **Web** 탭으로 이동합니다.
        - **Add a new web app**을 클릭하고, Flask를 선택합니다.
        - PythonAnywhere는 기본적으로 WSGI 파일을 생성합니다. 이를 수정하여 애플리케이션을 연결.

    - WSGI 파일 경로는 일반적으로 `/var/www/yourusername_pythonanywhere_com_wsgi.py`입니다. 이 파일을 열고 다음 내용으로 수정합니다:
    
        ```python
        import sys
        path = '/home/yourusername/your-repo-name'
        if path not in sys.path:
            sys.path.append(path)

        from app import app as application
        ```


### **2.5 애플리케이션 실행 및 테스트**
![http_request](/assets/images/http_request.jpg)

#### 1. PythonAnywhere의 웹 서버 구성
- Nginx:
    - 클라이언트의 웹 요청을 처리하고, uWSGI 서버로 요청을 전달합니다.
    - 정적 파일(이미지, CSS, JavaScript 등)을 효율적으로 제공합니다.
- uWSGI:
    - WSGI 프로토콜을 사용하여 Python 웹 애플리케이션(Flask, Django 등)을 실행합니다.
    - Nginx와 Python 웹 애플리케이션 사이의 통신을 담당합니다.

- 사용자 설정 제한
    - PythonAnywhere는 사용자가 Nginx 설정을 직접 변경할 수 없도록 제한합니다.
    - 이는 시스템의 안정성과 보안을 유지하기 위한 조치입니다.

#### 2. PythonAnywhere에서 Flask 앱 실행 방식
- WSGI 서버: 
    - PythonAnywhere는 WSGI (Web Server Gateway Interface) 서버를 사용하여 웹 애플리케이션을 실행합니다. 
    - WSGI 서버는 웹 서버와 웹 애플리케이션 간의 통신을 담당합니다.
- WSGI 파일: 
    - PythonAnywhere는 WSGI 설정 파일을 사용하여 Flask 애플리케이션을 로드하고 실행합니다. - WSGI 파일은 PythonAnywhere 웹 인터페이스의 "Web" 탭에서 설정할 수 있습니다.
- 애플리케이션 객체: 
    - WSGI 파일은 Flask 애플리케이션 객체를 로드하고 WSGI 서버에 전달합니다. 
    - PythonAnywhere는 이 애플리케이션 객체를 사용하여 웹 요청을 처리합니다.
- `with app.app_context()` 블록을 사용하여 애플리케이션 컨텍스트 안에서 init_db() 함수를 호출하면, 애플리케이션 시작 시 데이터베이스가 초기화됩니다.

    ```python
    with app.app_context():
        init_db()
    ```

#### 3. 앱 Reload
- Web 탭에서 **Reload** 버튼을 클릭하여 애플리케이션을 다시 시작합니다.
- 제공된 URL(예: `http://isbnone.pythonanywhere.com`)로 접속하여 애플리케이션이 정상적으로 실행되는지 확인합니다.


### **2.6 추가 설정 (옵션)**
- **환경 변수**: `.env` 파일이나 PythonAnywhere의 **Web** 탭에서 환경 변수를 설정할 수 있습니다.
- **데이터베이스**: MySQL 데이터베이스를 사용하는 경우 PythonAnywhere의 데이터베이스 설정을 확인.
