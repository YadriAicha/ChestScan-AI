
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# =========================
# 1. Configuration
# =========================
st.set_page_config(
    page_title="Chest X-Ray Classification",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 Chest X-Ray Classification")
st.write(
    "Chargez une radiographie thoracique pour obtenir "
    "une prédiction du modèle CNN."
)

# =========================
# 2. Charger le modèle
# =========================
@st.cache_resource
def load_model():
    return keras.models.load_model("chest_xray_cnn.keras")

try:
    model = load_model()
except Exception as e:
    st.error(
        "Impossible de charger le modèle. "
        "Vérifiez que chest_xray_cnn.keras est présent "
        "dans le dossier du projet."
    )
    st.stop()

# =========================
# 3. Importer une image
# =========================
uploaded_file = st.file_uploader(
    "Choisissez une radiographie",
    type=["jpg", "jpeg", "png"]
)

# =========================
# 4. Prétraiter l'image
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Radiographie chargée",
        use_container_width=True
    )

    if st.button("Analyser la radiographie"):

        # Redimensionnement
        image = image.resize((180, 180))

        # Conversion en tableau NumPy
        image_array = np.array(image, dtype=np.float32)

        # Normalisation entre 0 et 1
        image_array = image_array / 255.0

        # Ajouter la dimension du batch
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        prediction = model.predict(image_array, verbose=0)

        # Classification binaire avec une sortie sigmoid
        probability = float(np.asarray(prediction).reshape(-1)[0])

        # Adapter ce seuil à l'encodage des classes du dataset
        if probability >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = probability
        else:
            predicted_class = "NORMAL"
            confidence = 1 - probability

        # Affichage
        st.subheader("Résultat de la prédiction")

        if predicted_class == "PNEUMONIA":
            st.warning("Classe prédite : PNEUMONIA")
        else:
            st.success("Classe prédite : NORMAL")

        st.metric(
            "Score associé à la classe prédite",
            f"{confidence * 100:.2f} %"
        )

        st.caption(
            "Résultat expérimental d'un modèle de classification, "
            "ne constituant pas un diagnostic médical."
        )
