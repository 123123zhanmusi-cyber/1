import streamlit as st
from predict import predict_ecg
from PIL import Image

st.title("Apple Watch ECG AI诊断")

file = st.file_uploader(
"上传 Apple Watch 心电图图片",
type=["png","jpg","jpeg"]
)

if file:

    image = Image.open(file)

    st.image(image,caption="上传的ECG")

    with open("temp.png","wb") as f:
        f.write(file.getbuffer())

    result = predict_ecg("temp.png")

    if result == "正常心律":

        st.success("检测结果：正常")

    else:

        st.error("检测到异常: " + result)