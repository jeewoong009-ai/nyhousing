import streamlit as st
import pandas as pd
import joblib

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="뉴욕 주택 인공지능 예측", layout="wide")
st.title("🤖 뉴욕 집값 인공지능 예측기")

# 2. 인공지능 모델 및 데이터 불러오기
@st.cache_resource
def load_model():
    return joblib.load('nyc_rf_model_small.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('nyc_final_data.csv')

rf_model = load_model()
df = load_data()

# 3. 훈련 당시의 열(기둥) 구조 복원
df_encoded = pd.get_dummies(df, columns=['homeType'], drop_first=True)
model_columns = df_encoded.drop(columns=['price', 'zipcode', 'latitude', 'longitude']).columns

# 4. 사이드바: 동적 조건 슬라이더 (현실적인 제한 적용)
st.sidebar.header("원하는 집의 조건을 만들어보세요")

# 4-1. 방 개수를 가장 먼저 선택
bed = st.sidebar.slider("방 개수", 1, 5, 2)

# 4-2. 욕실 개수는 방 개수 + 1.5개까지만 가능하도록 제한
max_bath = float(bed + 1.5)
bath = st.sidebar.slider("욕실 개수", 1.0, max_bath, float(bed), 0.5)

# 4-3. 집 크기(sqft)도 방 개수에 비례하여 최소/최대 평수 제한
min_sqft = bed * 300
max_sqft = bed * 1200
living_area = st.sidebar.slider("집 크기 (sqft)", min_sqft, max_sqft, bed * 600)

# 4-4. 거리 변수는 독립적으로 유지
subway_dist = st.sidebar.slider("지하철역까지 거리 (km)", 0.1, 5.0, 1.0)
park_dist = st.sidebar.slider("센트럴 파크까지 거리 (km)", 0.1, 20.0, 5.0)

# 5. 빈 데이터프레임 만들고 사용자 입력값 집어넣기
input_data = pd.DataFrame(0, index=[0], columns=model_columns)

if 'bedrooms' in input_data.columns: input_data['bedrooms'] = bed
if 'bathrooms' in input_data.columns: input_data['bathrooms'] = bath
if 'livingArea' in input_data.columns: input_data['livingArea'] = living_area
if 'dist_to_central_park' in input_data.columns: input_data['dist_to_central_park'] = park_dist
if 'dist_to_nearest_subway' in input_data.columns: input_data['dist_to_nearest_subway'] = subway_dist

# 기본 집 유형을 Condo(아파트)로 고정
if 'homeType_Condo' in input_data.columns: input_data['homeType_Condo'] = 1

# 6. 인공지능 가격 예측 수행
predicted_price = rf_model.predict(input_data)[0]

# 7. 결과 출력
st.subheader("💡 인공지능 예측 결과")
st.metric(label="이 조건의 집, 적정 가격은?", value=f"${predicted_price:,.0f}")
