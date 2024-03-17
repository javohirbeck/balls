import streamlit as st
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
st.title('To\'p klassifikatsiya modeli')

# rasm yuklash
file = st.file_uploader('Rasm yuklang', type=['png', 'jpg'])

if file:
    st.image(file, width=200)
    # PIL converter
    img = PILImage.create(file)

    model = load_learner('ball_model.pkl')
    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')