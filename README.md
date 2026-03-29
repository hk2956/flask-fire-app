# Project#1: AI기반 산불 예측 서비스 개발

## 📁 파일 구조
```
fire_project/
├── step1_2_train.py          ← 단계 1(전처리) + 단계 2(모델 학습) 실행 파일
├── sanbul-pwa-flask.py       ← 단계 3 Flask 웹앱 메인 파일
├── requirements.txt          ← 단계 4 Render 배포용 패키지 목록
├── Procfile                  ← 단계 4 Render 배포용 실행 명령
├── render.yaml               ← 단계 4 Render 설정 파일
├── fires_model.keras         ← (학습 후 생성) Keras 모델
├── fires_pipeline.pkl        ← (학습 후 생성) 전처리 파이프라인
├── sanbul2district-divby100.csv  ← 데이터 파일 (직접 다운로드)
└── templates/
    ├── index.html            ← 홈 페이지
    ├── prediction.html       ← 입력 폼 페이지
    └── result.html           ← 예측 결과 페이지
```

---

## 🚀 실행 순서

### ✅ STEP 0: 데이터 다운로드
구글드라이브에서 `sanbul2district-divby100.csv`를 다운받아  
`fire_project/` 폴더에 넣으세요.

---

### ✅ STEP 1 & 2: 전처리 + 모델 학습

```bash
cd fire_project
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
python step1_2_train.py
```

실행 후 생성 파일:
- `fires_model.keras` — 학습된 Keras 모델
- `fires_pipeline.pkl` — 전처리 파이프라인
- `1_3_histograms.png` — 히스토그램
- `1_4_log_transform_comparison.png` — 로그 변환 비교
- `1_6_scatter_matrix.png` — Scatter Matrix
- `1_7_geo_plot.png` — 지역별 시각화
- `2_training_curve.png` — 학습 곡선

---

### ✅ STEP 3: Flask 웹앱 로컬 실행

```bash
pip install flask flask-bootstrap5 flask-wtf wtforms gunicorn
python sanbul-pwa-flask.py
```

브라우저에서 `http://localhost:5000` 접속

---

### ✅ STEP 4: Render 배포

1. [https://github.com](https://github.com) 에서 새 저장소 생성
2. 아래 파일들을 GitHub에 업로드:
   ```
   sanbul-pwa-flask.py
   requirements.txt
   Procfile
   render.yaml
   fires_model.keras
   fires_pipeline.pkl
   sanbul2district-divby100.csv
   templates/index.html
   templates/prediction.html
   templates/result.html
   ```
3. [https://render.com](https://render.com) 에서 회원가입 후 로그인
4. **New → Web Service** 클릭
5. GitHub 저장소 연결
6. 설정:
   - **Name**: flask-fire-app (원하는 이름)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn sanbul-pwa-flask:app`
7. **Create Web Service** 클릭 → 자동 빌드 & 배포
8. 배포 완료 후 URL 캡처 (제출용)

> ⚠️ Render 무료 플랜은 첫 요청 시 30초~1분 지연이 있을 수 있습니다.

---

## 📊 단계별 배점

| 단계 | 내용 | 배점 |
|------|------|------|
| 1 | 데이터 전처리 (1-1 ~ 1-9) | 45점 |
| 2 | Keras 모델 개발 | 15점 |
| 3 | Flask 웹앱 개발 | 25점 |
| 4 | Render 배포 | 15점 |
| **합계** | | **100점** |

---

## 🔑 주요 특성 설명

| 특성 | 설명 |
|------|------|
| longitude | 경기도 경도 구역 (1~7) |
| latitude | 경기도 위도 구역 (1~7) |
| month | 발생 월 (01-Jan ~ 12-Dec) |
| day | 요일 (00-sun ~ 06-sat, 07-hol) |
| avg_temp | 평균 기온 (°C) |
| max_temp | 최고 기온 (°C) |
| max_wind_speed | 최대 풍속 (m/s) |
| avg_wind | 평균 풍속 (m/s) |
| **burned_area** | **산불 면적 (m²) ← 예측 대상** |

> `burned_area`는 학습 시 `ln(burned_area+1)` 로그 변환 적용,  
> 예측 후 `exp(x)-1`로 역변환하여 실제 면적(m²)으로 출력합니다.

---

© Samkeun Kim, Hankyong National University
