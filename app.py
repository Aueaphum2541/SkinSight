import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2


st.set_page_config(page_title="SkinSight AI", layout="wide")
st.sidebar.title("🧠 SkinSight AI")
selection = st.sidebar.radio("Menu", [
    "🏠 Home", 
    "📤 Upload Image", 
    "🩺 Preprocessing", 
    "📌 Lesion Annotation", 
    "🔍 Predict",
    "🧬 Model Status", 
    "📊 Dataset Viewer", 
    "📚 Literature Insights"])

# Session state
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
    
if "model" not in st.session_state:
    st.session_state.model = None


# Home
if selection == "🏠 Home":
    st.title("SkinSight AI: Intelligent Skin Lesion Assessment")
    st.markdown("""
    This system aims to detect skin cancer early using deep learning models.
    
    🔍 **Cascaded Pipeline**:
    1. **U-Net**: For precise lesion segmentation.
    2. **CNN Classifiers**: (ResNet50, DenseNet121, MobileNet) to identify cancer types.

    🧴 **Skin Cancer Classes** (ISIC dataset):
    - Actinic Keratosis
    - Basal Cell Carcinoma
    - Dermatofibroma
    - Melanoma
    - Nevus
    - Pigmented Benign Keratosis
    - Seborrheic Keratosis
    - Squamous Cell Carcinoma
    - Vascular Lesion

    📦 *Model and AI tools are in development. UI ready for integration.*
    
    ---
    ![Sample](https://upload.wikimedia.org/wikipedia/commons/5/5f/Melanoma.jpg)
    *Example of melanoma skin lesion*
    """)

# Upload
elif selection == "📤 Upload Image":
    st.title("Upload Skin Lesion Images")
    uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        for img in uploaded_files:
            st.image(Image.open(img), caption=img.name, width=250)

# Preprocessing
elif selection == "🩺 Preprocessing":
    st.title("Image Preprocessing Preview")
    if not st.session_state.uploaded_images:
        st.warning("Upload images first.")
    else:
        for img in st.session_state.uploaded_images:
            image = Image.open(img)
            st.image(image, caption=f"Original: {img.name}", width=300)
            st.text("✔️ Resized to 224x224\n✔️ Normalized\n✔️ Augmentations: Flip / Rotation (optional)")

# Annotation
elif selection == "📌 Lesion Annotation":
    st.title("Annotate Lesion Area")
    if not st.session_state.uploaded_images:
        st.warning("Upload images first.")
    else:
        for img in st.session_state.uploaded_images:
            st.subheader(f"Annotate: {img.name}")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)", stroke_width=2, height=300, width=300,
                drawing_mode="rect", key=f"canvas_{img.name}")
            if canvas_result.json_data:
                st.json(canvas_result.json_data)

# Model Status
elif selection == "🧬 Model Status":
    st.title("AI Model Pipeline")
    if st.button("📥 Load AI Model (.h5)"):
        try:
            st.session_state.model = load_model("skin_cancer_model.h5")
            st.success("✅ Model loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")

elif selection == "🔍 Predict":
    st.title("Skin Lesion Prediction")
    if not st.session_state.uploaded_images:
        st.warning("Please upload images first.")
    elif st.session_state.model is None:
        st.warning("Please load the AI model from the 'Model Status' tab.")
    else:
        labels = [
            "Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", "Melanoma",
            "Nevus", "Pigmented Benign Keratosis", "Seborrheic Keratosis",
            "Squamous Cell Carcinoma", "Vascular Lesion"
        ]

        for img_file in st.session_state.uploaded_images:
            img = Image.open(img_file).convert("RGB").resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = st.session_state.model.predict(img_array)[0]
            class_idx = np.argmax(pred)
            confidence = pred[class_idx] * 100

            st.image(img, caption=f"Prediction: {labels[class_idx]} ({confidence:.2f}%)", width=300)
            st.markdown(f"🧾 **Confidence Scores**: {dict(zip(labels, [f'{p*100:.2f}%' for p in pred]))}")



# Dataset Viewer
elif selection == "📊 Dataset Viewer":
    st.title("Uploaded Images")
    if not st.session_state.uploaded_images:
        st.warning("Upload images first.")
    else:
        df = pd.DataFrame({
            "File Name": [img.name for img in st.session_state.uploaded_images],
            "Status": ["Pending" for _ in st.session_state.uploaded_images]
        })
        st.dataframe(df)

# Literature Insights
elif selection == "📚 Literature Insights":
    st.title("Key Papers in Skin Lesion Detection")
    with st.expander("🔄 Cascaded U-Net + CNN (Pattanaik et al., 2024)"):
        st.markdown("""
        - **U-Net for segmentation**, then **CNN (ResNet50/DenseNet121/MobileNet)** for classification.
        - Achieved **ResNet50: 96.6% accuracy** on ISIC dataset.
        - Handles class imbalance via feature extraction.
        """)
    with st.expander("📈 Transfer Learning with VGG / DenseNet"):
        st.markdown("- Pretrained models like VGG19 improve classification for small datasets.")
    with st.expander("🧠 DeepSkinNet / R-CNN"):
        st.markdown("- Specialized deep networks for melanoma detection using ISIC dataset.")

