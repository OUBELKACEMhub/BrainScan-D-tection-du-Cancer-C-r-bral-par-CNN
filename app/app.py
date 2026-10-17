"""
BrainScan AI - Application Streamlit simple et claire
Pour debutant : upload une IRM -> prediction
"""

import sys
import pickle
from pathlib import Path

import streamlit as st
from PIL import Image

# 1. Détection robuste de la racine du projet et ajout au sys.path
current_dir = Path(__file__).resolve().parent
if (current_dir.parent / "src").exists():
    PROJECT_ROOT = current_dir.parent
elif (current_dir / "src").exists():
    PROJECT_ROOT = current_dir
else:
    PROJECT_ROOT = Path("/app")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

from src.predict import predict_from_pil

st.set_page_config(page_title="BrainScan AI", page_icon="🧠", layout="centered")

st.title("🧠 BrainScan AI")
st.write("Classification des tumeurs cerebrales en 4 classes : **glioma, meningioma, notumor, pituitary**")
st.divider()


# 2. Chargement modele + encoder (dans la fonction mise en cache)
@st.cache_resource
def load_model_and_encoder():
    model = None
    le = None
    msg_model = ""
    msg_le = ""

    # Cherche le modele parmi tous les formats disponibles
    candidate_files = [
        MODELS_DIR / "final_model.keras",
        MODELS_DIR / "best_brainscan_model.keras",
        MODELS_DIR / "best_model.h5",
        MODELS_DIR / "brainscan_final.h5",
        Path("/app/models/final_model.keras"),
        Path("/app/models/best_model.h5"),
    ]
    last_error = ""
    for p in candidate_files:
        if p.exists():
            try:
                import tensorflow as tf

                # Wrapper pour supprimer 'quantization_config' ajouté par les versions récentes de Keras
                class CompatibleDense(tf.keras.layers.Dense):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("quantization_config", None)
                        super().__init__(*args, **kwargs)

                model = tf.keras.models.load_model(
                    str(p),
                    compile=False,
                    custom_objects={"Dense": CompatibleDense}
                )
                msg_model = f"✅ Modele charge : {p.name}"
                break
            except Exception as e:
                last_error = f"Erreur chargement {p.name} : {e}"

    if model is None:
        if last_error:
            msg_model = f"❌ {last_error}"
        else:
            msg_model = "Aucun modele trouve. Entraine d'abord via le notebook 03."

    # Cherche le label encoder
    le_path = MODELS_DIR / "label_encoder.pkl"
    if le_path.exists():
        try:
            with open(le_path, "rb") as f:
                le = pickle.load(f)
            msg_le = "Encoder charge"
        except Exception as e:
            msg_le = f"Erreur encoder : {e}"
            le = None

    # Fallback si pas d'encoder (pour demo sans entrainement)
    if le is None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["glioma", "meningioma", "notumor", "pituitary"])
        msg_le = "Encoder par defaut (glioma, meningioma, notumor, pituitary)"

    return model, le, msg_model, msg_le


model, le, msg_model, msg_le = load_model_and_encoder()

with st.sidebar:
    st.header("Info")
    st.write(msg_model)
    st.write(msg_le)
    st.divider()
    st.write("**Classes :**")
    st.write("- glioma")
    st.write("- meningioma")
    st.write("- notumor (pas de tumeur)")
    st.write("- pituitary")
    st.divider()
    st.caption("Entraine le modele : lance les notebooks 01 -> 05")

# Zone upload
st.subheader("1. Upload une image IRM")
uploaded = st.file_uploader("Choisis une image (jpg, png)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image chargee", width=350)

    if model is None:
        st.warning("Pas de modele disponible. Lance d'abord le notebook d'entrainement.")
    else:
        if st.button("Predire", type="primary"):
            with st.spinner("Prediction en cours..."):
                label, conf, proba = predict_from_pil(image, model, le)

            st.success(f"**Prediction : {label}**  -  Confiance : {conf:.1f}%")

            # Barres de proba
            st.write("Probabilites par classe :")
            for cls, p in proba.items():
                st.write(f"{cls} : {p:.1f}%")
                st.progress(int(p))

            # Petit conseil
            if label == "notumor":
                st.info("Aucune tumeur detectee.")
            else:
                st.warning(f"Tumeur detectee : {label}. Ceci est une aide, pas un diagnostic medical.")
else:
    st.info("Upload une image pour commencer.")

st.divider()
st.caption("Projet simple pour debutant - notebooks + Streamlit")