import streamlit as st
import numpy as np
import cv2
from PIL import Image
from feature_extraction import (
    remove_hair_and_preprocess,
    segment_lesion,
    extract_abcd_features
)

st.set_page_config(page_title="DermaScan Demo", page_icon="🔬", layout="centered")

st.title("🔬 DermaScan - Test")
st.write("Sistem de pre-screening al leziunilor cutanate")

# Alegere metoda de intrare: galerie sau camera telefonului
tab_upload, tab_camera = st.tabs(["📁 Încarcă fotografie", "📸 Fă o poză cu camera"])

imagine_sursa = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Apasă aici pentru a încărca fotografia",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        imagine_sursa = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    camera_file = st.camera_input("Încadrează alunița cât mai clar")
    if camera_file is not None:
        imagine_sursa = Image.open(camera_file).convert("RGB")

if imagine_sursa is not None:
    # Conversie din format PIL (RGB) in format OpenCV (BGR)
    img_rgb = np.array(imagine_sursa)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Pasul 1: Preprocesare, segmentare si extragere trasaturi ABCD
    with st.spinner("Se analizează imaginea..."):
        img_clean_bgr = remove_hair_and_preprocess(img_bgr)
        mask = segment_lesion(img_clean_bgr)
        features_vector, abcd_info = extract_abcd_features(img_clean_bgr, mask)

        # Desenam conturul detectat peste imaginea curatata (pentru vizualizare)
        img_contour_bgr = img_clean_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contour_bgr, contours, -1, (0, 255, 0), 2)

        # Conversie inapoi in RGB pentru afisare in Streamlit
        img_clean_rgb = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2RGB)
        img_contour_rgb = cv2.cvtColor(img_contour_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("1. Etapele de procesare a imaginii")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="Originală", use_container_width=True)
    with col2:
        st.image(img_clean_rgb, caption="Fără fire de păr (Morfologie)", use_container_width=True)
    with col3:
        st.image(mask, caption="Mască Otsu", use_container_width=True)

    st.image(img_contour_rgb, caption="Conturul leziunii detectate", use_container_width=True)

    st.subheader("2. Indicatori extrași (Regula ABCD)")
    if abcd_info:
        c1, c2 = st.columns(2)
        c1.metric("A - Asimetrie (Excentricitate)", abcd_info["Asimetrie (Excentricitate)"])
        c2.metric("B - Bordură (Compactitate)", abcd_info["Bordură (Compactitate)"])
        
        c3, c4 = st.columns(2)
        c3.metric("C - Culoare (Variație HSV)", abcd_info["Culoare (Variație HSV)"])
        c4.metric("D - Diametru echivalent (px)", abcd_info["Diametru echivalent (px)"])
    else:
        st.warning("Nu s-a putut izola clar leziunea. Încearcă o fotografie cu iluminare mai uniformă.")

    st.divider()
    st.subheader("3. Decizie AI (CNN + SVM)")
    st.info("⏳ Următorul pas: conectarea modelului antrenat pe Kaggle pentru clasificarea Benign vs. Malign.")