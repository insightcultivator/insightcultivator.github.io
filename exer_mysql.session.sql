-- 현재 연결된 데이터베이스 확인
SELECT SCHEMA();

-- students 테이블 생성
CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    grade CHAR(1),
    score INT
);

INSERT INTO students (id, name, grade, score) VALUES 
(1, 'Alice', 'A', 95),
(2, 'Bob', 'B', 85),
(3, 'Charlie', 'A', 92),
(4, 'David', 'C', 78),
(5, 'Eve', 'B', 88),
(6, 'Frank', 'A', 95),
(7, 'Grace', 'C', 72),
(8, 'Hannah', 'B', 85);

-- 3.1 SELECT & FROM

-- 모든 학생의 이름과 점수를 조회
SELECT name, score
FROM students;

-- 3.2 Comparison 연산자 & WHERE
SELECT name, score FROM students WHERE score >= 90;


-- 3.3 ORDER BY
SELECT name, score 
FROM students
ORDER BY score DESC;

-- 3.4 AND 조건
SELECT name, grade, score 
FROM students
WHERE grade = 'A' AND score >= 90;

-- 3.5 OR 조건
SELECT name, grade, score 
FROM students
WHERE grade = 'A' OR score >= 85;

-- 3.6 DISTINCT 
SELECT DISTINCT grade 
FROM students;

DROP TABLE students;

CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    grade CHAR(1)
);

-- 초기 데이터 삽입
INSERT INTO students (student_id, name, age, grade) VALUES
(1, 'Alice', 20, 'A'),
(2, 'Bob', 22, 'B'),
(3, 'Charlie', 21, 'C');

-- 4.1 INSERT 문
-- 새로운 학생 추가
INSERT INTO students (student_id, name, age, grade) VALUES
(4, 'David', 19, 'A');

-- 결과 확인
SELECT * FROM students;

-- 4.2 UPDATE 문
UPDATE students
SET grade = 'A'
WHERE student_id = 2;

-- 4.3 DELETE 문
-- Charlie의 데이터 삭제
DELETE FROM students
WHERE student_id = 3;

-- 4.4 TRUNCATE TABLE 문
-- 테이블의 모든 데이터 삭제
TRUNCATE TABLE students;


-- SELECT * FROM students;

DROP TABLE students;
--students 테이블 생성
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    grade CHAR(1),
    course_id INT,
    email VARCHAR(100)
);


-- course 테이블 생성
CREATE TABLE courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(50),
    instructor VARCHAR(50)
);

-- students 테이블에 데이터 삽입
INSERT INTO students (student_id, name, age, grade, course_id, email) VALUES
(1, 'Alice', 20, 'A',101, 'alice@example.com'),
(2, 'Bob', 22, 'B', 102, NULL),
(3, 'Charlie', 19, 'C',NULL, 'charlie@example.com'),
(4, 'David', 21, 'A', 101, 'david@example.com'),
(5, 'Eve', 22, 'F', NULL, NULL);

-- courses 테이블에 데이터 삽입
INSERT INTO courses (course_id, course_name, instructor) VALUES
(101, 'Mathmatics', 'Dr. Smith'),
(102, 'Physics', 'Dr. Johnson'),
(103, 'Chemistry', 'Dr. Brown');

-- 5.1 NOT 조건
-- 수학 과목을 듣지 않는 학생 목록을 조회
SELECT * FROM students
WHERE course_id IS NULL OR course_id NOT IN (SELECT course_id FROM courses WHERE course_name = 'Mathmatics');

-- 5.2 ALIASES
-- 학생 이름과 이메일을 조회하며, 열 이름을 다르게 표시
SELECT name AS 'Student Name', email AS 'Email Address'
FROM students;

-- 5.3 JOINS
-- 학생 이름과 그들이 수강 중인 과목 이름 및 강사를 조회
SELECT s.name AS "Student Name" , c.course_name AS "Course Name", c.instructor AS "Instructor"
FROM students s 
JOIN courses c ON s.course_id = c.course_id;

-- 5.4 BETWEEN 조건
-- 나이가 20세에서 22사이인 학생 목록을 조회
SELECT *
FROM students
WHERE age BETWEEN 20 AND 22;

-- 5.5 IN 조건
-- 성적이 'A' 또는 'B'인 학생 목록을 조회
SELECT *
FROM students
WHERE grade IN('A','B')

-- 5.6 IS NULL 조건
-- 이메일 주소가 없는 학생 목록을 조회
SELECT * 
FROM students
WHERE email IS NULL;

-- 5.7 IS NOT NULL
-- 이메일 주소가 있는 학생을 조회
SELECT *
FROM students
WHERE email is NOT NULL;

-- 5.8 LIKE 조건
-- 이름이 'A'로 시작하는 학생 목록을 조회
SELECT *
FROM students
WHERE name LIKE 'A%' ;

DROP TABLE students;


-- 6.고급 쿼리

-- 학생 정보 테이블
CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    Name VARCHAR(50),
    Age INT,
    Major VARCHAR(50)
);

DROP TABLE Courses;
-- 과목 정보 테이블
CREATE TABLE Courses(
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(50),
    Credits INT
);

-- 수강 정보 테이블(학생과 과목의 관계)
CREATE TABLE Enrollments (
    EnrollmentID INT PRIMARY KEY,
    StudentID INT,
    CourseID INT,
    Grade CHAR(1)
);

-- 데이터 삽입
INSERT INTO Students(StudentID, Name, Age, Major) VALUES
(1, 'Alice', 20, 'Computer Science'),
(2, 'Bob', 22, 'Mathmatics'),
(3, 'Charlie', 21, 'Physics'),
(4, 'David', 23, 'Chemistry');

INSERT INTO Courses(CourseID, CourseName, Credits) VALUES
(101, 'Database Systems', 3),
(102, 'Calculus', 4),
(103, 'Quantum Mechanics', 3),
(104, 'Organic Chemistry', 4);

INSERT INTO Enrollments (EnrollmentID, StudentID, CourseID, Grade) VALUES
(1001, 1,101, 'A'),
(1002, 1,102, 'B'),
(1003, 2,102, 'A'),
(1004, 3,103, 'C'),
(1005, 4,104, 'B');


-- 6.1 EXISTS 조건
-- 서브쿼리의 결과가 존재하는지 확인
SELECT Name
FROM Students
WHERE EXISTS(
    SELECT 1
    FROM Enrollments e
    JOIN Courses c ON e.CourseID = c.CourseID
    WHERE e.StudentID = Students.StudentID AND c.CourseName = 'Database Systems'
);

-- 6.2 GROUP BY 절
-- 특정 열을 기준으로 데이터를 그룹화
-- 각 전공별로 몇 명의 학생이 있는지 확인
SELECT Major, COUNT(*) AS StudentCount
FROM Students
GROUP BY Major;

-- 6.3 HAVING 절
-- 그룹화된 데이터에 조건을 적용
-- 수강한 과목 수가 2개 이상인 학생 조회
SELECT StudentID, COUNT(*) AS CourseCount
FROM Enrollments
GROUP BY StudentID
HAVING COUNT(*) >=2 

-- 6.4 SELECT LIMIT 문
-- 검색결과의 개수를 제한
-- 나이가 많은 순으로 상위 2명의 학생 조회
SELECT Name, Age
FROM Students
ORDER BY Age DESC
LIMIT 2;

-- 6.5 DELETE LIMIT 문
-- 삭제하는 데이터의 개수를 제한
-- 가장 낮은 학점(C)을 받은 수강 기록 중 하나 삭제
DELETE FROM Enrollments
WHERE Grade='C'
LIMIT 1;

SELECT * FROM Enrollments;

-- 6.6 UNION 연산자
-- 여러 SELECT 문의 결과를 합쳐서 보여준다(중복제거)

SELECT Name FROM Students WHERE Major = 'Computer Science'
UNION
SELECT Name FROM Students WHERE Major = 'Physics';

-- 6.7 UNION ALL 연산자
-- 여러 SELECT 문의 결과를 합쳐서 보여준다.(중복포함)
SELECT Name FROM Students WHERE Major = 'Computer Science'
UNION ALL
SELECT Name FROM Students WHERE Major = 'Physics';

-- 6.8 INTERSECT 연산자
SELECT StudentID
FROM Enrollments
WHERE CourseID = 101 AND StudentID IN (
    SELECT StudentID
    FROM Enrollments
    WHERE CourseID = 102
);

SELECT DISTINCT e1.StudentID
FROM Enrollments e1
JOIN Enrollments e2
ON e1.StudentID = e2.StudentID
WHERE e1.CourseID = 101 AND e2.CourseID = 102;

-- Subqueries
-- 쿼리 안에 또 다른 쿼리를 사용하여 복잡한 검색을 수행
-- 평균 수강 과목 수보다 많은 과목을 수강한 학생 조회
SELECT StudentID, COUNT(*) AS CourseCount
FROM Enrollments
GROUP BY StudentID
HAVING COUNT(*) > (
    SELECT AVG(CourseCount)
    FROM (
        SELECT COUNT(*) CourseCount
        FROM Enrollments
        GROUP BY StudentID
    ) AS AvgCourses
);

  -- 테이블 삭제 (선택사항)
  DROP TABLE Courses, Enrollments;

--   7. 테이블 관리

-- 7.1 Data Types 및 CREATE TABLE 문
-- 테이블 열의 데이터 유형을 정의하고, 새로운 테이블을 생성
-- 학생 정보를 저장할 테이블 생성
CREATE TABLE Students(
    StudentID INT PRIMARY KEY, -- 정수형 기본키
    Name VARCHAR(50),  -- 가변길이 문자열 (최대 50자)
    Age INT,  -- 정수형
    Email VARCHAR(100),  -- 이메일 주소(가변 길이 문자열)
    EnrollmentDate DATE -- 날짜 형식
);

INSERT INTO Students(StudentID, Name, Age, Email, EnrollmentDate)VALUES
(1,'Alice', 20, 'alice@example.com','2023-09-01'),
(2,'Bob', 22, 'bob@example.com','2023-09-02');

SELECT * FROM Students;

-- 7.2 CREATE TABLE AS 문
-- 기존 테이블의 데이터를 기반으로 새로운 테이블을 생성
-- Students 테이블에서 나이가 21세 이상인 학생들만 포함하는 새 테이블 생성
CREATE TABLE AdultStudents AS
SELECT * FROM Students WHERE Age >= 21;

SELECT * FROM AdultStudents;

-- 7.3 Primary KEys
-- 테이브르이 각 행을 고유하게 식별하는 열을 정의
-- 강의 정보를 저장할 테이블 생성(CourseID 기본 키)
CREATE TABLE Courses(
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(100),
    Credits INT
);

-- 기본키를 사용하여 데이터를 삽입하고 중복된 CourseID를 삽이하려고 시도
INSERT INTO Courses(CourseID, CourseName, Credits) VALUES 
(101, 'Mathmatics', 3),
(102, 'Physics', 4);

INSERT INTO Courses(CourseID, CourseName, Credits) VALUES 
(101, 'Chemistry', 3);

-- 7.4 ALTER TABLE 문
-- 테이블의 구조를 변경

-- Students 테이블에 전화번호 열 추가
ALTER TABLE Students ADD PhoneNumber VARCHAR(15);

-- Students 테이블에 PhoneNumber 열 삭제
ALTER TABLE Students DROP PhoneNumber;

-- Students 테이블의 Email 열 이름 변경
ALTER TABLE Students RENAME COLUMN Email TO ContactEmail;

DESCRIBE Students;

-- 7.5 DROP TABLE문
-- 테이블을 삭제
DROP TABLE Enrollments;

SHOW TABLES;

-- 7.6 VIEW
-- 하나 이상의 테이블을 기반으로 가상의 테이블을 생성

  -- Enrollments 테이블 생성
  CREATE TABLE Enrollments (
      EnrollmentID INT PRIMARY KEY,
      StudentID INT,
      CourseID INT,
      FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
      FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
  );

  -- 데이터 삽입
  INSERT INTO Enrollments (EnrollmentID, StudentID, CourseID) VALUES
  (1, 1, 101), -- Alice가 Database Systems 수강
  (2, 2, 102); -- Bob이 Operating Systems 수강
  

DROP VIEW StudentCourseView;
SELECT * FROM students;
SELECT * FROM courses;
SELECT * FROM Enrollments;

  -- 뷰 생성
  CREATE VIEW StudentCourseView AS 
  SELECT s.StudentID, s.Name AS StudentName, c.CourseName, c.Credits
  FROM Students s
  JOIN Enrollments e ON s.StudentID = e.StudentID
  JOIN Courses c ON e.CourseID = c.CourseID;

  -- 뷰 조회

  SELECT * FROM Courses;

--   7.7 Unique Constraints
-- 열의 값이 중복되지 않도록 제약 조건을 설정
ALTER TABLE Courses ADD CONSTRAINT UniqueCourseName UNIQUE (CourseName);

INSERT INTO Courses (CourseID, CourseName, Credits)
VALUES (103, 'Physics', 4);

-- 7.8 Indexes
-- 데이터 검색 속도를 향상시키기 위한 인덱스를 생성
-- Students 테이블의 Name열에 인덱스 생성

EXPLAIN SELECT * FROM Students WHERE Name = 'Alice';
CREATE INDEX idx_student_name On Students(Name);
EXPLAIN SELECT * FROM Students WHERE Name = 'Alice';

-- 8. 사용자 및 권한
-- 환경 설정 및 데이터 준비
CREATE DATABASE IF NOT EXISTS school;
USE school;

CREATE TABLE students(
    student_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    grade INT NOT NULL
);

-- 샘플 데이터 삽입
INSERT INTO students (name, grade) VALUES
('Alice', 1),
('Bob',2),
('Charlie',3);

SELECT * FROM students;


-- 8.1 CREATE USER문
-- 새로운 사용자를 생성합니다. 
CREATE USER 'teacher'@'localhost' IDENTIFIED BY 'password123';

-- 8.2 Grant/Revoke Privileges
-- 사용자에게 특정 권한을 부여하거나 취소
-- teacher사용자에게 students 테이블에 대한 SELECT 권한 부여
GRANT SELECT ON school.students TO 'teacher'@'localhost';

-- teacher 사용자에게 INSERT 권한 추가 부여
GRANT INSERT ON school.students TO 'teacher'@'localhost';

-- 8.3 Show grants for user in MySQL
-- 특정 사용자의 현재 권한을 확인합니다.
-- teacher 사용자의 권한 확인
SHOW GRANTS FOR 'teacher'@'localhost';

-- teacher 사용자의 이름을 instructor로 변경
RENAME USER 'teacher'@'localhost' TO 'instructor'@'localhost';

-- 8.6 DROP USER문
-- 사용자를 삭제
DROP USER 'instructor'@'localhost';

-- 8.7 Find users logged into MySql
-- 현재 접속한 사용자 목록 확인
SELECT USER(), CURRENT_USER();
SHOW PROCESSLIST;

-- 실습 후 정리
DROP USER IF EXISTS 'teacher'@'localhost';
DROP USER IF EXISTS 'instructor'@'localhost';
DROP DATABASE IF EXISTS school;


-- 9. 프로그래밍 요소
USE SchoolDB;


CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    enrollment_date DATE NOT NULL
);

INSERT INTO students(name, age, enrollment_date) VALUES
('Alice', 20, '2024-01-15'),
('Bob', 22, '2023-09-01'),
('Charlie', 21, '2022-06-20');

SELECT * FROM students;

-- 9.1 Comments within SQL
SELECT * FROM students;

/* 여러 줄 예제
특정 학생 정보를 가져오는 쿼리입니다. */
SELECT * FROM students WHERE name = 'Alice';

-- 9.2 Literals
-- 문자열, 숫자, 날짜 등의 고정된 값을 표현하는 데이터
SELECT 'HELLO, SQL!' AS text_literal;
SELECT 123 AS number_literal;
SELECT '2025-01-01' AS date_literal;

-- 9.3 Declaring Variables
-- SQL 내에서 값을 저장하고 활용하기 위해 변수를 선언하고 할당
-- MySQL에서는 SET을 사용하여 변수를 선언하고 값을 할당
SET @student_name = 'David';
SET @student_age = 23;

SELECT @student_name AS name, @student_age AS age;

-- 9.4 Sequences(AUTO_INCREMENT)
-- 테이블의 기본키 값을 자동으로 증가
INSERT INTO students (name, age, enrollment_date) VALUES
('David', 23, '2024-02-01');

-- 마지막 삽입된 ID 확인
SELECT LAST_INSERT_ID();

-- 9.5 DELIMTER와 BEGIN..END
-- 저장 프로시저나 함수, 트리거 작성시 사용
-- DELIMITER는 이 구분자를 임시로 변경하는 것
-- BEGIN..END는 여러개의 SQL 문장을 하나의 블록으로 묶어서 실행
DROP PROCEDURE IF EXISTS AddNumbers;
DROP FUNCTION IF EXISTS GetStudentAge;
DROP FUNCTION IF EXISTS GetStudentGrdade;

-- SQLTools에서 DELIMITER 에러 발생, WorkBench에서 작업해야
-- SQLTools에서 Procedure, Funtions, Trigger가 UI에 표시안됨. 확인은 가능

DELIMITER $$
CREATE PROCEDURE AddNumbers(IN num1 INT, IN num2 INT, OUT result INT)
BEGIN
    SET result = num1 + num2;
END $$
DELIMITER ;


SET @num1 = 10;
SET @num2 = 20;
SET @result = 0;

-- 저장 프로시저 호출
CALL AddNumbers(@num1, @num2, @result);

SELECT @result AS Result;


-- procedure 및 Funtions 확인 코드
SELECT ROUTINE_NAME, ROUTINE_TYPE 
FROM INFORMATION_SCHEMA.ROUTINES 
WHERE ROUTINE_SCHEMA = 'school';

-- 9.6 Furntions
-- 실습할 테이블 생성
CREATE TABLE products (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL,
        price DECIMAL(10, 2) NOT NULL,
        category VARCHAR(50)
    );

INSERT INTO products (name, price, category) VALUES
('노트북', 1200.00, '전자제품'),
('마우스', 30.00, '전자제품'),
('셔츠', 50.00, '의류'),
('바지', 70.00, '의류'),
('책', 20.00, '도서');

SELECT * FROM products;

-- 사용자 정의 함수
DELIMITER //

CREATE FUNCTION calculate_discounted_price(price DECIMAL(10, 2), discount_rate DECIMAL(3, 2))
RETURNS DECIMAL(10, 2)
DETERMINISTIC
BEGIN
    DECLARE discounted_price DECIMAL(10, 2);
    SET discounted_price = price * (1 - discount_rate);
    RETURN discounted_price;
END //

DELIMITER ;

-- 생성된 ROUTINE 확인
SELECT ROUTINE_NAME, ROUTINE_TYPE 
FROM INFORMATION_SCHEMA.ROUTINES 
WHERE ROUTINE_SCHEMA = 'school';


-- 함수 호출 예제
SELECT name, price, calculate_discounted_price(price, 0.1) AS discounted_price
FROM products;

-- 9.7 프로시저

-- students 테이블 생성
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT,
    grade VARCHAR(50)
);

-- 샘플 데이터 삽입
INSERT INTO students (name, age, grade) VALUES
('김철수', 18, 'A'),
('박영희', 19, 'B'),
('이민수', 18, 'A'),
('정수진', 20, 'C'),
('최지훈', 19, 'B');

-- 특정 학년 학생 정보 조회 프로시저
DELIMITER //

CREATE PROCEDURE GetStudentsByGrade(IN grade_to_find VARCHAR(50))
BEGIN
    SELECT * FROM students WHERE grade = grade_to_find;
END //

DELIMITER ;

-- 프로시저 실행
CALL GetStudentsByGrade('B');

-- 9.8 IF-THEN-ELSEIF-ELSE-END 문
-- 샘플 테이블 생성
  CREATE TABLE IF NOT EXISTS scores (
      id INT AUTO_INCREMENT PRIMARY KEY,
      student_name VARCHAR(255) NOT NULL,
      score INT
  );

  INSERT INTO scores (student_name, score) VALUES
  ('김철수', 85),
  ('박영희', 92),
  ('이민수', 78),
  ('정수진', 65),
  ('최지훈', 50);
  
--   점수에 따라 학점을 부여하는 AssignGrade 프로시저 생성
  DELIMITER //

  CREATE PROCEDURE AssignGrade(IN student_id INT)
  BEGIN
      DECLARE student_score INT;
      DECLARE student_grade VARCHAR(2);

      -- 학생 점수 조회
      SELECT score INTO student_score FROM scores WHERE id = student_id;

      -- 학점 부여
      IF student_score >= 90 THEN
          SET student_grade = 'A';
      ELSEIF student_score >= 80 THEN
          SET student_grade = 'B';
      ELSEIF student_score >= 70 THEN
          SET student_grade = 'C';
      ELSEIF student_score >= 60 THEN
          SET student_grade = 'D';
      ELSE
          SET student_grade = 'F';
      END IF;

      -- 결과 출력
      SELECT student_name, student_score, student_grade FROM scores WHERE id = student_id;
  END //

  DELIMITER ;


-- 생성된 ROUTINE 확인
SELECT ROUTINE_NAME, ROUTINE_TYPE 
FROM INFORMATION_SCHEMA.ROUTINES 
WHERE ROUTINE_SCHEMA = 'school';

  -- 실행 예제
  CALL AssignGrade(1); 

--   9.9 WHILE 문
  DELIMITER //

  CREATE PROCEDURE PrintNumbers()
  BEGIN
      DECLARE counter INT DEFAULT 1;

      WHILE counter <= 5 DO
          SELECT counter AS number;
          SET counter = counter + 1;
      END WHILE;
  END //

  DELIMITER ;

  -- 실행 예제
  CALL PrintNumbers();

--  9.10 LEAVE 문
  DELIMITER //

  CREATE PROCEDURE TestLeave()
  BEGIN
      DECLARE counter INT DEFAULT 1;
        
      test_loop: LOOP
          IF counter > 3 THEN
              LEAVE test_loop;
          END IF;
          SELECT counter AS number;
          SET counter = counter + 1;
      END LOOP;
  END //

  DELIMITER ;

  -- 실행 예제
  CALL TestLeave();

--   9.11 ITERATE문(특정 부분 건너뛰기)
  DELIMITER //

  CREATE PROCEDURE TestIterate()
  BEGIN
      DECLARE counter INT DEFAULT 0;
        
      test_loop: LOOP
          SET counter = counter + 1;
            
          IF counter = 2 THEN
              ITERATE test_loop; -- 2일 때는 출력하지 않고 다음 반복으로 이동
          END IF;

          SELECT counter AS number;
            
          IF counter >= 5 THEN
              LEAVE test_loop;
          END IF;
      END LOOP;
  END //

  DELIMITER ;

  -- 실행 예제
  CALL TestIterate();

--   9.12 RETURN문
  DELIMITER //

  CREATE FUNCTION GetTotalStudents() RETURNS INT DETERMINISTIC
  BEGIN
      DECLARE total INT;
      SELECT COUNT(*) INTO total FROM students;
      RETURN total;
  END //

  DELIMITER ;

  -- 실행 예제
  SELECT GetTotalStudents() AS Total_Students;

--   9.13 LOOP문

SELECT ROUTINE_NAME, ROUTINE_TYPE
FROM INFORMATION_SCHEMA.ROUTINES
WHERE ROUTINE_SCHEMA = 'school';

  DELIMITER //

  CREATE PROCEDURE TestLoop()
  BEGIN
      DECLARE counter INT DEFAULT 1;
        
      simple_loop: LOOP
          SELECT counter AS number;
          SET counter = counter + 1;
          IF counter > 3 THEN
              LEAVE simple_loop;
          END IF;
      END LOOP;
  END //

  DELIMITER ;

  -- 실행 예제
  CALL TestLoop();

--   9.14 REPEAT 문
  DELIMITER //

  CREATE PROCEDURE TestRepeat()
  BEGIN
      DECLARE counter INT DEFAULT 1;
        
      REPEAT
          SELECT counter AS number;
          SET counter = counter + 1;
      UNTIL counter > 3 END REPEAT;
  END //

  DELIMITER ;

  -- 실행 예제
  CALL TestRepeat();

--   9.15 CASE문
 UPDATE students SET age = 28 WHERE id = 1;
 UPDATE students SET age = 25 WHERE id = 5;
  SELECT name, 
      age,
      CASE 
          WHEN age < 21 THEN '미성년자'
          WHEN age BETWEEN 21 AND 25 THEN '청년'
          ELSE '성인'
      END AS category
  FROM students;



CREATE DATABASE IF NOT EXISTS SchoolDB;
USE SchoolDB;

CREATE TABLE Students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    grade CHAR(1)
);

INSERT INTO Students (name, age, grade) VALUES 
('Alice', 14, 'A'),
('Bob', 15, 'B'),
('Charlie', 16, 'A'),
('David', 14, 'C'),
('Eve', 15, 'B');


--   9.16 Cursor 선언 및 핸들링
-- SQL에서 여러 행을 하나씩 처리할 수 있도록 커서를 선언
-- 커서 선언 (학생 정보를 조회하는 쿼리)
DECLARE studentCursor CURSOR FOR 
SELECT name, grade FROM Students;
-- 전체 코드는 밑에 있음

USE schoolDB;






DELIMITER $$

CREATE PROCEDURE FetchStudents()
BEGIN
    DECLARE done INT DEFAULT FALSE;  -- NOT FOUND 핸들러용 변수
    DECLARE studentName VARCHAR(50); -- 학생 이름 저장 변수
    DECLARE studentGrade CHAR(1);    -- 학생 등급 저장 변수
    
    -- 커서 선언 (학생 정보를 조회하는 쿼리)
    DECLARE studentCursor CURSOR FOR 
    SELECT name, grade FROM Students;
    
    -- NOT FOUND 상황 처리 핸들러
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    -- 커서 열기
    OPEN studentCursor;

    -- 반복문으로 한 행씩 데이터 가져오기
    read_loop: LOOP
        FETCH studentCursor INTO studentName, studentGrade;

        IF done THEN
            LEAVE read_loop;
        END IF;

        -- 가져온 데이터 출력
        SELECT CONCAT('학생: ', studentName, ', 등급: ', studentGrade) AS Student_Info;
    END LOOP;

    -- 커서 닫기
    CLOSE studentCursor;
END $$

DELIMITER ;

-- 실습 실행방법
  CALL FetchStudents();