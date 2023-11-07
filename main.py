import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time

import torch

# VGG 모델 불러오기
vgg_model = load_model('model_vgg_2500.h5')

# YOLO 모델 불러오기
yolo_model = load_model('model_yolo.pt')

# 타이틀
st.markdown("<h1 style='text-align: center; color: green;'>그린</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'> <span style='color: green;'>G</span>eenery <span style='color: green;'>R</span>ecovering A<span style='color: green;'>I</span> solutio<span style='color: green;'>N</span></h1>", unsafe_allow_html=True)


# 여백 추가
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 이미지 업로드 위젯 추가
uploaded_images = st.file_uploader("<이미지 업로드>", type=["jpg", "png", "jpeg", "tif"], accept_multiple_files=True)

# 여백 추가
st.markdown("<br>", unsafe_allow_html=True)

# radio 버튼 추가 및 중앙 정렬
image_selection = st.radio("탐지 유형을 선택하세요:", ("정상 지역", "불법 산림 의심 지역", "불법 산림 의심 지역 객체 탐지"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# AI 버튼 중앙 정렬
st.markdown("<style>.stButton button {display: block; margin: 0 auto; width: 200px; height: 50px; font-size: 20px;}</style>", unsafe_allow_html=True)

if st.button('AI 탐지'):
    if uploaded_images:
        with st.spinner('AI 탐지 중...'):
            # 이미지 목록 생성
            image_list = [Image.open(image) for image in uploaded_images]

            # 업로드한 이미지 파일 이름 목록
            image_names = [image.name for image in uploaded_images]

            for img, img_name in zip(image_list, image_names):
                img = img.resize((280, 280))

                img_array = image.img_to_array(img)
                img_array = img_array[:, :, :3]
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                prediction = vgg_model.predict(img_array)

                if image_selection == "정상 지역" and prediction > 0.5:
                    st.success(image_selection)
                    st.image(img, caption=f"이미지 이름: {img_name}", use_column_width=True)
                    
                elif image_selection == "불법 산림 의심 지역" and prediction <= 0.5:
                    st.error(image_selection)
                    st.image(img, caption=f"이미지 이름: {img_name}", use_column_width=True)
                    
                elif image_selection == "불법 산림 의심 지역 객체 탐지" and prediction <= 0.5:
                    st.success("객체 탐지 이미지")
                    results = yolo_model(img)
                    # 결과 이미지를 Pillow Image로 변환
                    result_image = results.render()[0]
                    # Pillow Image를 NumPy 배열로 변환
                    result_np = np.array(result_image)
                    
                    # 이미지를 Streamlit 앱에서 표시
                    st.image(result_np, caption=f"이미지 이름: {img_name}", use_column_width=True)

            # 로딩 시간 조절
            time.sleep(2)