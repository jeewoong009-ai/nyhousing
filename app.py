import streamlit as st
import pandas as pd
import joblib

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="뉴욕 주택 인공지능 예측", layout="wide")
st.title("🤖 뉴욕 집값 인공지능 예측기")
st.write("당신이 원하는 집의 조건을 입력하면, 인공지능이 뉴욕의 적정 가격을 예측합니다.")

# 2. 인공지능 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load('nyc_rf_model_small.pkl')

rf_model = load_model()

# 3. 사이드바: 가상의 집 조건 입력받기
st.sidebar.header("원하는 집의 조건을 만들어보세요")
bed = st.sidebar.slider("방 개수", 1, 5, 2)
bath = st.sidebar.slider("욕실 개수", 1.0, 5.0, 1.0, 0.5)
living_area = st.sidebar.slider("집 크기 (sqft)", 500, 5000, 1500)
subway_dist = st.sidebar.slider("지하철역까지 거리 (km)", 0.1, 5.0, 1.0)
park_dist = st.sidebar.slider("센트럴 파크까지 거리 (km)", 0.1, 20.0, 5.0)

# 4. 모델이 학습했던 원본 열 이름 그대로 가져오기
model_columns = rf_model.feature_names_in_

# 5. 모든 값을 0으로 채운 빈 데이터프레임 만들기 (모델 형식에 딱 맞춤)
input_data = pd.DataFrame(0, index=[0], columns=model_columns)

# 6. 사용자가 슬라이더로 입력한 값만 쏙쏙 집어넣기
if 'bedrooms' in input_data.columns: input_data['bedrooms'] = bed
if 'bathrooms' in input_data.columns: input_data['bathrooms'] = bath
if 'livingArea' in input_data.columns: input_data['livingArea'] = living_area
if 'dist_to_central_park' in input_data.columns: input_data['dist_to_central_park'] = park_dist
if 'dist_to_nearest_subway' in input_data.columns: input_data['dist_to_nearest_subway'] = subway_dist

# (기본 집 유형을 Condo로 가정하여 1을 넣어줌)
if 'homeType_Condo' in input_data.columns: input_data['homeType_Condo'] = 1

# 7. 인공지능 가격 예측 수행
predicted_price = rf_model.predict(input_data)[0]

# 8. 결과 출력
st.subheader("💡 인공지능 예측 결과")
st.metric(label="이 조건의 집, 적정 가격은?", value=f"${predicted_price:,.0f}")

st.write("---")
st.write("이 예측은 2021년 실제 시장 데이터를 바탕으로 인공지능이 학습한 결과입니다.")
