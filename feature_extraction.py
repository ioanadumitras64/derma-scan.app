import cv2
import numpy as np
from skimage.measure import label, regionprops

def remove_hair_and_preprocess(img_bgr, size=(224, 224)):
    # 1. Redimensionare (apib3)
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Eliminarea firelor de par prin morfologie Black-Hat (apib4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Reconstructie (inpainting) peste firele de par
    img_clean = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # 3. Filtrare mediana pentru reducerea zgomotului (apib3)
    img_clean = cv2.medianBlur(img_clean, 3)
    return img_clean

def segment_lesion(img_clean):
    # Conversie in grayscale si egalizare adaptiva a contrastului (apib2)
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Prag automat Otsu - alunita e mai inchisa la culoare (apib8)
    _, mask = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Curatare morfologica: inchidere si deschidere (apib4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Pastram doar componenta conexa cea mai mare (apib6)
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(mask)
    
    largest_region = max(regions, key=lambda r: r.area)
    clean_mask = (labeled == largest_region.label).astype(np.uint8) * 255
    return clean_mask

def extract_abcd_features(img_clean, mask):
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros(16), {}
    
    prop = regions[0]
    area = prop.area
    perimeter = prop.perimeter if prop.perimeter > 0 else 1.0
    
    # B - Bordura: Compactitate / Circularitate (apib6_1)
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    
    # A - Asimetrie: Excentricitate si soliditate (apib6_1, apib6_2)
    eccentricity = prop.eccentricity
    solidity = prop.solidity
    
    # D - Diametru echivalent si axa majora (apib6_1)
    equiv_diameter = prop.equivalent_diameter_area
    major_axis = prop.major_axis_length
    
    # C - Culoare: Medie si deviatie standard pe canalele RGB si HSV (apib1)
    hsv = cv2.cvtColor(img_clean, cv2.COLOR_BGR2HSV)
    lesion_pixels_rgb = img_clean[mask == 255]
    lesion_pixels_hsv = hsv[mask == 255]
    
    if len(lesion_pixels_rgb) == 0:
        return np.zeros(16), {}
        
    rgb_mean = np.mean(lesion_pixels_rgb, axis=0)
    rgb_std = np.std(lesion_pixels_rgb, axis=0)
    hsv_std = np.std(lesion_pixels_hsv, axis=0)
    
    # Vectorul numeric complet (pentru modelul SVM din Pasul 2)
    features_vector = np.hstack([
        area, perimeter, compactness, eccentricity, solidity,
        equiv_diameter, major_axis,
        rgb_mean, rgb_std, hsv_std
    ])
    
    # Dictionar lizibil (pentru afisare in interfata Streamlit)
    abcd_dict = {
        "Asimetrie (Excentricitate)": round(float(eccentricity), 3),
        "Bordură (Compactitate)": round(float(compactness), 3),
        "Culoare (Variație HSV)": round(float(np.mean(hsv_std)), 2),
        "Diametru echivalent (px)": round(float(equiv_diameter), 1)
    }
    
    return features_vector, abcd_dict