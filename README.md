# travelmate-model
추천 모델 관련 fast API 서버

## 레포지토리 구조

```
travelmate-model/
├── app/
│ ├── init.py
│ ├── crud.py
│ ├── database.py
│ ├── main.py
│ ├── models.py
│ ├── preprocessing.py
│ ├── recommendation.py
│ └── schema.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```


## 테스트 명령어
- python -m venv venv : 가상환경 'venv'라는 이름으로 생성
- source venv/bin/activate   ( Windows에서는 `venv\Scripts\activate` 사용 ) : 만든 가상환경 활성화
- pip install -r requirements.txt : 필요한 종속성 설치

- 루트 디렉토리에 .env 파일 추가하기

- uvicorn app.main:app --reload : 서버 실행 (기본 8000번 포트에서 실행)
