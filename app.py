
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------- CONFIGURATION ----------------

MODEL_PATH = "deepfake_model.keras"
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.50
MAX_VIDEO_FRAMES = 30
MAX_UPLOAD_MB = 200

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- CUSTOM UI ----------------

st.title("🔍 Deepfake Detection System")
st.write(
    "Analyze an image or video using a trained deep learning model. "
    "Results are estimates and should not be treated as proof of authenticity."
)

st.sidebar.header("Detection Settings")

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.30,
    max_value=0.70,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help=(
        "Scores at or above this threshold are classified as REAL. "
        "Changing this threshold does not retrain the model."
    ),
)

max_frames = st.sidebar.slider(
    "Video frames to analyze",
    min_value=5,
    max_value=MAX_VIDEO_FRAMES,
    value=20,
    step=5,
)

st.sidebar.caption(
    "More frames can improve coverage but increase processing time."
)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Model not found: {MODEL_PATH}. "
        "Place deepfake_model.keras in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ---------------- PREPROCESSING ----------------

def preprocess_image(image):
    """Prepare an image using the MobileNetV2 preprocessing convention."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)

    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def get_scores(real_scores):
    """
    Assumes model output is a sigmoid score where larger means REAL.
    Confirm this matches the model's training labels.
    """
    scores = np.asarray(real_scores, dtype=np.float32).reshape(-1)

    if len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction scores.")

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(
            "Model output is outside [0, 1]. "
            "Check whether your model uses a sigmoid output."
        )

    real_score = float(np.mean(scores))
    fake_score = 1.0 - real_score

    label = "REAL" if real_score >= threshold else "FAKE"

    # This is the winning score, not a calibrated probability.
    winning_score = max(real_score, fake_score)

    return {
        "label": label,
        "real_score": real_score,
        "fake_score": fake_score,
        "winning_score": winning_score,
    }


# ---------------- IMAGE PREDICTION ----------------

def predict_image(image):
    arr = preprocess_image(image)
    prediction = model.predict(arr, verbose=0)

    if prediction.size != 1:
        raise ValueError(
            "Expected one sigmoid output. Check your model architecture."
        )

    result = get_scores([float(prediction.reshape(-1)[0])])
    result["frames"] = 1

    return result


# ---------------- VIDEO PREDICTION ----------------

def predict_video(path, frame_limit=20):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open this video. Try a supported MP4 video."
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("The video contains no readable frames.")

        positions = np.linspace(
            0,
            total_frames - 1,
            min(frame_limit, total_frames),
            dtype=int,
        )

        scores = []
        frame_numbers = []

        progress = st.progress(0, text="Preparing video analysis...")

        for index, position in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()

            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, IMG_SIZE)

                arr = frame.astype(np.float32)
                arr = np.expand_dims(arr, axis=0)

                arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                    arr
                )

                prediction = model.predict(arr, verbose=0)

                if prediction.size == 1:
                    score = float(prediction.reshape(-1)[0])

                    if np.isfinite(score) and 0 <= score <= 1:
                        scores.append(score)
                        frame_numbers.append(int(position) + 1)

            progress.progress(
                (index + 1) / len(positions),
                text=f"Analyzing frame {index + 1} of {len(positions)}",
            )

        progress.empty()

        if not scores:
            raise ValueError("No frames could be analyzed successfully.")

        result = get_scores(scores)
        result["frames"] = len(scores)
        result["frame_scores"] = scores
        result["frame_numbers"] = frame_numbers
        result["total_frames"] = total_frames

        return result

    finally:
        cap.release()


# ---------------- RESULT DISPLAY ----------------

def display_result(result):
    label = result["label"]

    if label == "REAL":
        st.success("Prediction: REAL")
    else:
        st.error("Prediction: FAKE")

    col1, col2, col3 = st.columns(3)

    col1.metric("Real score", f'{result["real_score"] * 100:.2f}%')
    col2.metric("Fake score", f'{result["fake_score"] * 100:.2f}%')
    col3.metric("Winning model score", f'{result["winning_score"] * 100:.2f}%')

    chart_data = pd.DataFrame(
        {
            "Class": ["REAL", "FAKE"],
            "Score (%)": [
                result["real_score"] * 100,
                result["fake_score"] * 100,
            ],
        }
    ).set_index("Class")

    st.bar_chart(chart_data, y="Score (%)")

    st.caption(
        f"Classification threshold: {threshold:.2f}. "
        "Model scores are not necessarily calibrated probabilities."
    )


# ---------------- FILE UPLOAD ----------------

upload_type = st.radio(
    "Choose media type",
    ["Image", "Video"],
    horizontal=True,
)

if upload_type == "Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Image is larger than {MAX_UPLOAD_MB} MB.")
            st.stop()

        try:
            image = Image.open(uploaded)
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                width=400,
            )

            if st.button("Analyze image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = predict_image(image)

                display_result(result)

        except (UnidentifiedImageError, OSError):
            st.error("The uploaded image is invalid or corrupted.")
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

else:
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Video is larger than {MAX_UPLOAD_MB} MB.")
            st.stop()

        st.video(uploaded)

        if st.button("Analyze video", type="primary"):
            temp_path = None

            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp:
                    temp.write(uploaded.getvalue())
                    temp_path = temp.name

                with st.spinner("Analyzing sampled video frames..."):
                    result = predict_video(
                        temp_path,
                        frame_limit=max_frames,
                    )

                display_result(result)

                st.write(
                    f'**Frames analyzed:** {result["frames"]} '
                    f'out of {result["total_frames"]} total frames'
                )

                frame_data = pd.DataFrame(
                    {
                        "Frame": result["frame_numbers"],
                        "Real score (%)": [
                            score * 100
                            for score in result["frame_scores"]
                        ],
                    }
                ).set_index("Frame")

                st.subheader("Frame-by-frame analysis")
                st.line_chart(frame_data, y="Real score (%)")

                st.caption(
                    "The video prediction aggregates sampled frame scores. "
                    "This model does not directly analyze temporal motion, "
                    "and a frame-level score can be unreliable on its own."
                )

            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass


# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Research prototype only. Results may be incorrect, especially for "
    "unseen datasets, low-quality media, compression, or unusual lighting. "
    "Do not use this output alone to make important decisions."
)
