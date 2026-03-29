"""
Project#1: AI기반 산불 예측 서비스 개발
단계 1: 데이터 전처리 (45점)
단계 2: Keras 모델 개발 (15점)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 GUI 없이 사용
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 단계 1-1: Data 불러오기
# =====================================================================
print("\n" + "="*70)
print("단계 1-1: Data 불러오기")
print("="*70)

fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")

print("원본 데이터 (로그 변환 전):")
print(fires.head())
print(f"\n데이터 shape: {fires.shape}")

# =====================================================================
# 단계 1-4: 로그 변환 (1-3 시각화보다 먼저 적용 - 비교용)
# =====================================================================
# 원본 burned_area 히스토그램을 위해 백업
fires_original = fires.copy()

# 로그 변환 적용
fires['burned_area'] = np.log(fires['burned_area'] + 1)
print("\n로그 변환 후 데이터:")
print(fires.head())


# =====================================================================
# 단계 1-2: fires.head(), fires.info(), fires.describe(),
#            카테고리형 특성 month, day에 대해 value_counts() 출력
# =====================================================================
print("\n" + "="*70)
print("단계 1-2: 기본 데이터 정보 출력")
print("="*70)

print("\n### fires.head():")
print(fires.head())

print("\n### fires.info():")
fires.info()

print("\n### fires.describe():")
print(fires.describe())

print("\nMonth category value_counts:")
print(fires["month"].value_counts())

print("\nDay category value_counts:")
print(fires["day"].value_counts())


# =====================================================================
# 단계 1-3: 데이터 시각화 (히스토그램)
# =====================================================================
print("\n" + "="*70)
print("단계 1-3: 데이터 시각화")
print("="*70)

fires.hist(bins=50, figsize=(15, 10))
plt.suptitle("Feature Histograms", fontsize=14)
plt.tight_layout()
plt.savefig("1_3_histograms.png", dpi=100)
plt.close()
print("히스토그램 저장: 1_3_histograms.png")

# 두 속성 비교 (avg_temp vs burned_area)
plt.figure(figsize=(8, 5))
plt.scatter(fires["avg_temp"], fires["burned_area"], alpha=0.4, color='steelblue')
plt.xlabel("avg_temp")
plt.ylabel("burned_area (log)")
plt.title("avg_temp vs burned_area")
plt.tight_layout()
plt.savefig("1_3_scatter_temp_area.png", dpi=100)
plt.close()
print("속성 비교 산점도 저장: 1_3_scatter_temp_area.png")


# =====================================================================
# 단계 1-4: burned_area 로그 변환 히스토그램 비교
# =====================================================================
print("\n" + "="*70)
print("단계 1-4: burned_area 로그 변환 비교 히스토그램")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 변환 전
axes[0].hist(fires_original["burned_area"], bins=50, color='steelblue')
axes[0].set_title("burned_area (원본)")
axes[0].set_xlabel("burned_area")

# 변환 후
axes[1].hist(fires["burned_area"], bins=50, color='orange')
axes[1].set_title("burned_area (log 변환 후)")
axes[1].set_xlabel("ln(burned_area + 1)")

plt.suptitle("Log Transformation Effect", fontsize=13)
plt.tight_layout()
plt.savefig("1_4_log_transform_comparison.png", dpi=100)
plt.close()
print("로그 변환 비교 히스토그램 저장: 1_4_log_transform_comparison.png")


# =====================================================================
# 단계 1-5: train_test_split + StratifiedShuffleSplit
# =====================================================================
print("\n" + "="*70)
print("단계 1-5: Train/Test Set 분리")
print("="*70)

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
print(f"train_test_split 결과 - train: {len(train_set)}, test: {len(test_set)}")
print(f"Test set 비율: {len(test_set)/len(fires)*100:.1f}%")
print("\ntest_set.head():")
print(test_set.head())

# StratifiedShuffleSplit (month 기준 층화 샘플링)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set  = fires.loc[test_index]

print("\nMonth category proportion (Stratified test set):")
print(strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion:")
print(fires["month"].value_counts() / len(fires))


# =====================================================================
# 단계 1-6: Pandas scatter_matrix() - 4개 이상 특성
# =====================================================================
print("\n" + "="*70)
print("단계 1-6: Scatter Matrix")
print("="*70)

from pandas.plotting import scatter_matrix

attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attributes], figsize=(10, 8), alpha=0.3)
plt.suptitle("Scatter Matrix (4 features)", fontsize=13)
plt.tight_layout()
plt.savefig("1_6_scatter_matrix.png", dpi=100)
plt.close()
print("Scatter Matrix 저장: 1_6_scatter_matrix.png")


# =====================================================================
# 단계 1-7: 지역별 burned_area plot
#            (s=max_temp, c=burned_area)
# =====================================================================
print("\n" + "="*70)
print("단계 1-7: 지역별 burned_area 지도 시각화")
print("="*70)

fires.plot(kind="scatter", x="longitude", y="latitude",
           alpha=0.4,
           s=fires["max_temp"] * 3,   # 원 크기 = max_temp
           label="max_temp",
           c="burned_area",
           cmap=plt.get_cmap("jet"),
           colorbar=True,
           figsize=(10, 7))
plt.title("Gyeonggi-do Forest Fire: size=max_temp, color=burned_area")
plt.legend()
plt.tight_layout()
plt.savefig("1_7_geo_plot.png", dpi=100)
plt.close()
print("지역별 시각화 저장: 1_7_geo_plot.png")


# =====================================================================
# 단계 1-8: OneHotEncoder - month, day 인코딩
# =====================================================================
print("\n" + "="*70)
print("단계 1-8: OneHotEncoder 인코딩")
print("="*70)

from sklearn.preprocessing import OneHotEncoder

# training set 준비
fires_features = strat_train_set.drop(["burned_area"], axis=1)
fires_labels   = strat_train_set["burned_area"].copy()

fires_num = fires_features.drop(["month", "day"], axis=1)

# month 인코딩
cat_month = fires_features[["month"]]
cat_month_encoder = OneHotEncoder()
fires_month_1hot = cat_month_encoder.fit_transform(cat_month)
print("cat_month_encoder.categories_:")
print(cat_month_encoder.categories_)
print(f"month 인코딩 shape: {fires_month_1hot.shape}")

# day 인코딩
cat_day = fires_features[["day"]]
cat_day_encoder = OneHotEncoder()
fires_day_1hot  = cat_day_encoder.fit_transform(cat_day)
print("\ncat_day_encoder.categories_:")
print(cat_day_encoder.categories_)
print(f"day 인코딩 shape: {fires_day_1hot.shape}")


# =====================================================================
# 단계 1-9: Pipeline + StandardScaler로 training set 생성
# =====================================================================
print("\n" + "="*70)
print("단계 1-9: Pipeline + StandardScaler로 전처리")
print("="*70)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires_features)
print(f"전처리된 training set shape: {fires_prepared.shape}")
print(f"(행: {fires_prepared.shape[0]}개 샘플, 열: {fires_prepared.shape[1]}개 특성)")

# test set도 동일하게 전처리
fires_test_features = strat_test_set.drop(["burned_area"], axis=1)
fires_test_labels   = strat_test_set["burned_area"].copy()
fires_test_prepared = full_pipeline.transform(fires_test_features)
print(f"전처리된 test set shape: {fires_test_prepared.shape}")


# =====================================================================
# 단계 2: Keras 모델 개발 (Regression MLP)
# =====================================================================
print("\n" + "="*70)
print("단계 2: Keras Regression MLP 모델 개발")
print("="*70)

import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# 추가 validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42
)
X_test  = fires_test_prepared
y_test  = fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

# 모델 구성: 3개 은닉층 (각 30 뉴런, ReLU) + 출력층 1개
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid),
    verbose=0  # 훈련 로그 생략 (200 에포크)
)
print("모델 훈련 완료 (200 epochs)")

# 학습 곡선 저장
plt.figure(figsize=(10, 4))
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig("2_training_curve.png", dpi=100)
plt.close()
print("학습 곡선 저장: 2_training_curve.png")

# 모델 저장
model.save('fires_model.keras')
print("모델 저장 완료: fires_model.keras")

# 모델 평가
mse_test = model.evaluate(X_test, y_test, verbose=0)
rmse_test = np.sqrt(mse_test)
print(f"\n[Test Set 평가]")
print(f"  MSE  : {mse_test:.4f}")
print(f"  RMSE : {rmse_test:.4f}")

# 예측 결과 확인 (처음 3개)
X_new = X_test[:3]
predictions_log = model.predict(X_new, verbose=0)
predictions_area = np.expm1(predictions_log)   # 역변환: e^x - 1

print("\nnp.round(model.predict(X_new), 2):")
print(np.round(predictions_log, 2))

print("\n역변환된 예측 면적 (m²):")
for i, area in enumerate(predictions_area.flatten()):
    print(f"  샘플 {i+1}: {area:.2f} m²")

# Pipeline과 함께 저장 (Flask에서 사용)
import joblib
joblib.dump(full_pipeline, 'fires_pipeline.pkl')
print("\nPipeline 저장 완료: fires_pipeline.pkl")

print("\n" + "="*70)
print("단계 1 & 2 완료!")
print("="*70)
