import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "deepfake_model.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Deepfake Detection System")
st.write(
    "Upload an image or video to estimate whether it appears real or fake."
)
st.warning(
    "This is an educational ML model. Predictions may be incorrect "
    "and should not be treated as proof of authenticity."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Put deepfake_model.keras beside app.py.")
    st.stop()

model = load_model()


def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def predict_image(image):
    arr = preprocess_image(image)
    real_score = float(model.predict(arr, verbose=0)[0][0])
    fake_score = 1.0 - real_score

    return {
        "label": "REAL" if real_score >= 0.5 else "FAKE",
        "real_score": real_score,
        "fake_score": fake_score,
    }


def predict_video(path, max_frames=20):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError("Unable to read this video. Try an H.264 MP4.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError("This video has no readable frames.")

    positions = np.linspace(
        0, total - 1, min(max_frames, total), dtype=int
    )

    scores = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, IMG_SIZE)
        arr = np.expand_dims(frame.astype(np.float32), axis=0)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

        scores.append(float(model.predict(arr, verbose=0)[0][0]))

    cap.release()

    if not scores:
        raise ValueError("No frames could be analyzed.")

    real_score = float(np.mean(scores))

    return {
        "label": "REAL" if real_score >= 0.5 else "FAKE",
        "real_score": real_score,
        "fake_score": 1.0 - real_score,
        "frames": len(scores),
        "frame_scores": scores,
    }


upload_type = st.radio(
    "Choose upload type",
    ["Image", "Video"],
    horizontal=True
)

if upload_type == "Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=400)

        if st.button("Analyze image"):
            with st.spinner("Analyzing image..."):
                result = predict_image(image)

            st.subheader("Prediction: " + result["label"])
            st.metric("Real score", f'{result["real_score"] * 100:.2f}%')
            st.metric("Fake score", f'{result["fake_score"] * 100:.2f}%')

            st.bar_chart({
                "Score": {
                    "Real": result["real_score"],
                    "Fake": result["fake_score"]
                }
            })

else:
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is not None:
        if st.button("Analyze video"):
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp:
                temp.write(uploaded.getvalue())
                temp_path = temp.name

            try:
                with st.spinner("Analyzing sampled video frames..."):
                    result = predict_video(temp_path)

                st.subheader("Prediction: " + result["label"])
                st.metric("Real score", f'{result["real_score"] * 100:.2f}%')
                st.metric("Fake score", f'{result["fake_score"] * 100:.2f}%')
                st.write("Frames analyzed:", result["frames"])

                st.bar_chart({
                    "Score": {
                        "Real": result["real_score"],
                        "Fake": result["fake_score"]
                    }
                })

                st.caption(
                    "Video scores are averages of sampled frame predictions. "
                    "This model does not analyze temporal motion."
                )

            except Exception as exc:
                st.error(f"Could not analyze video: {exc}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)