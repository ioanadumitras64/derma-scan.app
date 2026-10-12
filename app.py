import streamlit as st
from PIL import Image

st.set_page_config(page_title="DermaScan Demo", page_icon="🔬", layout="centered")

st.title("🔬 DermaScan - Test")
st.write("Sistem de pre-screening al leziunilor cutanate")

#upload imagine sau fotografiere
uploaded_file = st.file_uploader(
    "Apasă aici pentru a încărca fotografia", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    imagine = Image.open(uploaded_file)
    st.image(imagine, caption="Imagine încărcată cu succes", use_container_width=True)
