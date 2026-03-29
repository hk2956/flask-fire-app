"""
Project#1: AI기반 산불 예측 서비스 개발
단계 3: Flask Web App (25점)
"""

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
import joblib
import os

from flask import Flask, render_template, request, redirect, url_for
# from bootstrap_flask import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)

# =====================================================================
# Flask 앱 초기화
# =====================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
# bootstrap5 = Bootstrap5(app)

# =====================================================================
# 모델 & 파이프라인 로드
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = keras.models.load_model(os.path.join(BASE_DIR, 'fires_model.keras'))
pipeline = joblib.load(os.path.join(BASE_DIR, 'fires_pipeline.pkl'))

print("모델 & 파이프라인 로드 완료")

# =====================================================================
# WTForms 정의
# =====================================================================
MONTH_CHOICES = [
    ('01-Jan', '01-Jan'), ('02-Feb', '02-Feb'), ('03-Mar', '03-Mar'),
    ('04-Apr', '04-Apr'), ('05-May', '05-May'), ('06-Jun', '06-Jun'),
    ('07-Jul', '07-Jul'), ('08-Aug', '08-Aug'), ('09-Sep', '09-Sep'),
    ('10-Oct', '10-Oct'), ('11-Nov', '11-Nov'), ('12-Dec', '12-Dec'),
]
DAY_CHOICES = [
    ('00-sun', '00-sun (일)'), ('01-mon', '01-mon (월)'), ('02-tue', '02-tue (화)'),
    ('03-wed', '03-wed (수)'), ('04-thu', '04-thu (목)'), ('05-fri', '05-fri (금)'),
    ('06-sat', '06-sat (토)'), ('07-hol', '07-hol (공휴일)'),
]

class LabForm(FlaskForm):
    longitude      = StringField('longitude (1~7, 경기도 구역)', validators=[DataRequired()])
    latitude       = StringField('latitude (1~7, 경기도 구역)',  validators=[DataRequired()])
    month          = SelectField('month', choices=MONTH_CHOICES)
    day            = SelectField('day',   choices=DAY_CHOICES)
    avg_temp       = StringField('avg_temp (평균기온 °C)',  validators=[DataRequired()])
    max_temp       = StringField('max_temp (최고기온 °C)',  validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed (최대풍속 m/s)', validators=[DataRequired()])
    avg_wind       = StringField('avg_wind (평균풍속 m/s)', validators=[DataRequired()])
    submit         = SubmitField('Submit')


# =====================================================================
# 라우트
# =====================================================================
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()

    if form.validate_on_submit():
        # 폼에서 입력값 수집
        longitude      = float(form.longitude.data)
        latitude       = float(form.latitude.data)
        month          = form.month.data
        day            = form.day.data
        avg_temp       = float(form.avg_temp.data)
        max_temp       = float(form.max_temp.data)
        max_wind_speed = float(form.max_wind_speed.data)
        avg_wind       = float(form.avg_wind.data)

        # DataFrame 생성
        num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp',
                        'max_wind_speed', 'avg_wind']
        cat_attribs = ['month', 'day']

        input_data = pd.DataFrame([{
            'longitude'      : longitude,
            'latitude'       : latitude,
            'month'          : month,
            'day'            : day,
            'avg_temp'       : avg_temp,
            'max_temp'       : max_temp,
            'max_wind_speed' : max_wind_speed,
            'avg_wind'       : avg_wind,
        }])

        # 전처리 (저장된 pipeline 사용)
        input_prepared = pipeline.transform(input_data)

        # 예측 (로그 스케일) → 역변환
        pred_log  = model.predict(input_prepared, verbose=0)
        pred_area = float(np.expm1(pred_log[0][0]))
        pred_area = max(pred_area, 0.0)   # 음수 방지

        return render_template(
            'result.html',
            pred_area   = round(pred_area, 2),
            longitude   = longitude,
            latitude    = latitude,
            month       = month,
            day         = day,
            avg_temp    = avg_temp,
            max_temp    = max_temp,
            max_wind_speed = max_wind_speed,
            avg_wind    = avg_wind,
        )

    return render_template('prediction.html', form=form)


# =====================================================================
# 실행
# =====================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
