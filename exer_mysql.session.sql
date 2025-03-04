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
  
  SELECT * FROM StudentCourseView;