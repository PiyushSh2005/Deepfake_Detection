
# import os
# import tempfile

# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf

# from PIL import Image, ImageOps, UnidentifiedImageError


# # ---------------- CONFIGURATION ----------------

# MODEL_PATH = "deepfake_model.keras"
# IMG_SIZE = (224, 224)
# DEFAULT_THRESHOLD = 0.50
# MAX_VIDEO_FRAMES = 30
# MAX_UPLOAD_MB = 200

# st.set_page_config(
#     page_title="Deepfake Detection System",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # ---------------- CUSTOM UI ----------------

# st.title("🔍 Deepfake Detection System")
# st.write(
#     "Analyze an image or video using a trained deep learning model. "
#     "Results are estimates and should not be treated as proof of authenticity."
# )

# st.sidebar.header("Detection Settings")

# threshold = st.sidebar.slider(
#     "Decision threshold",
#     min_value=0.30,
#     max_value=0.70,
#     value=DEFAULT_THRESHOLD,
#     step=0.05,
#     help=(
#         "Scores at or above this threshold are classified as REAL. "
#         "Changing this threshold does not retrain the model."
#     ),
# )

# max_frames = st.sidebar.slider(
#     "Video frames to analyze",
#     min_value=5,
#     max_value=MAX_VIDEO_FRAMES,
#     value=20,
#     step=5,
# )

# st.sidebar.caption(
#     "More frames can improve coverage but increase processing time."
# )


# # ---------------- LOAD MODEL ----------------

# @st.cache_resource
# def load_model(model_path):
#     return tf.keras.models.load_model(model_path, compile=False)


# if not os.path.isfile(MODEL_PATH):
#     st.error(
#         f"Model not found: {MODEL_PATH}. "
#         "Place deepfake_model.keras in the same folder as app.py."
#     )
#     st.stop()

# try:
#     model = load_model(MODEL_PATH)
# except Exception as exc:
#     st.error(f"Could not load the model: {exc}")
#     st.stop()


# # ---------------- PREPROCESSING ----------------

# def preprocess_image(image):
#     """Prepare an image using the MobileNetV2 preprocessing convention."""
#     image = ImageOps.exif_transpose(image).convert("RGB")
#     image = image.resize(IMG_SIZE)

#     arr = np.asarray(image, dtype=np.float32)
#     arr = np.expand_dims(arr, axis=0)

#     return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


# def get_scores(real_scores):
#     """
#     Assumes model output is a sigmoid score where larger means REAL.
#     Confirm this matches the model's training labels.
#     """
#     scores = np.asarray(real_scores, dtype=np.float32).reshape(-1)

#     if len(scores) == 0 or not np.all(np.isfinite(scores)):
#         raise ValueError("The model returned invalid prediction scores.")

#     if np.any(scores < 0) or np.any(scores > 1):
#         raise ValueError(
#             "Model output is outside [0, 1]. "
#             "Check whether your model uses a sigmoid output."
#         )

#     real_score = float(np.mean(scores))
#     fake_score = 1.0 - real_score

#     label = "REAL" if real_score >= threshold else "FAKE"

#     # This is the winning score, not a calibrated probability.
#     winning_score = max(real_score, fake_score)

#     return {
#         "label": label,
#         "real_score": real_score,
#         "fake_score": fake_score,
#         "winning_score": winning_score,
#     }


# # ---------------- IMAGE PREDICTION ----------------

# def predict_image(image):
#     arr = preprocess_image(image)
#     prediction = model.predict(arr, verbose=0)

#     if prediction.size != 1:
#         raise ValueError(
#             "Expected one sigmoid output. Check your model architecture."
#         )

#     result = get_scores([float(prediction.reshape(-1)[0])])
#     result["frames"] = 1

#     return result


# # ---------------- VIDEO PREDICTION ----------------

# def predict_video(path, frame_limit=20):
#     cap = cv2.VideoCapture(path)

#     if not cap.isOpened():
#         raise ValueError(
#             "Unable to open this video. Try a supported MP4 video."
#         )

#     try:
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         if total_frames <= 0:
#             raise ValueError("The video contains no readable frames.")

#         positions = np.linspace(
#             0,
#             total_frames - 1,
#             min(frame_limit, total_frames),
#             dtype=int,
#         )

#         scores = []
#         frame_numbers = []

#         progress = st.progress(0, text="Preparing video analysis...")

#         for index, position in enumerate(positions):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
#             ok, frame = cap.read()

#             if ok and frame is not None:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, IMG_SIZE)

#                 arr = frame.astype(np.float32)
#                 arr = np.expand_dims(arr, axis=0)

#                 arr = tf.keras.applications.mobilenet_v2.preprocess_input(
#                     arr
#                 )

#                 prediction = model.predict(arr, verbose=0)

#                 if prediction.size == 1:
#                     score = float(prediction.reshape(-1)[0])

#                     if np.isfinite(score) and 0 <= score <= 1:
#                         scores.append(score)
#                         frame_numbers.append(int(position) + 1)

#             progress.progress(
#                 (index + 1) / len(positions),
#                 text=f"Analyzing frame {index + 1} of {len(positions)}",
#             )

#         progress.empty()

#         if not scores:
#             raise ValueError("No frames could be analyzed successfully.")

#         result = get_scores(scores)
#         result["frames"] = len(scores)
#         result["frame_scores"] = scores
#         result["frame_numbers"] = frame_numbers
#         result["total_frames"] = total_frames

#         return result

#     finally:
#         cap.release()


# # ---------------- RESULT DISPLAY ----------------

# def display_result(result):
#     label = result["label"]

#     if label == "REAL":
#         st.success("Prediction: REAL")
#     else:
#         st.error("Prediction: FAKE")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Real score", f'{result["real_score"] * 100:.2f}%')
#     col2.metric("Fake score", f'{result["fake_score"] * 100:.2f}%')
#     col3.metric("Winning model score", f'{result["winning_score"] * 100:.2f}%')

#     chart_data = pd.DataFrame(
#         {
#             "Class": ["REAL", "FAKE"],
#             "Score (%)": [
#                 result["real_score"] * 100,
#                 result["fake_score"] * 100,
#             ],
#         }
#     ).set_index("Class")

#     st.bar_chart(chart_data, y="Score (%)")

#     st.caption(
#         f"Classification threshold: {threshold:.2f}. "
#         "Model scores are not necessarily calibrated probabilities."
#     )


# # ---------------- FILE UPLOAD ----------------

# upload_type = st.radio(
#     "Choose media type",
#     ["Image", "Video"],
#     horizontal=True,
# )

# if upload_type == "Image":
#     uploaded = st.file_uploader(
#         "Upload an image",
#         type=["jpg", "jpeg", "png", "webp"],
#         key="image_upload",
#     )

#     if uploaded is not None:
#         if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
#             st.error(f"Image is larger than {MAX_UPLOAD_MB} MB.")
#             st.stop()

#         try:
#             image = Image.open(uploaded)
#             image.load()
#             image = ImageOps.exif_transpose(image).convert("RGB")

#             st.image(
#                 image,
#                 caption="Uploaded image",
#                 width=400,
#             )

#             if st.button("Analyze image", type="primary"):
#                 with st.spinner("Analyzing image..."):
#                     result = predict_image(image)

#                 display_result(result)

#         except (UnidentifiedImageError, OSError):
#             st.error("The uploaded image is invalid or corrupted.")
#         except Exception as exc:
#             st.error(f"Image analysis failed: {exc}")

# else:
#     uploaded = st.file_uploader(
#         "Upload a video",
#         type=["mp4", "avi", "mov", "mkv"],
#         key="video_upload",
#     )

#     if uploaded is not None:
#         if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
#             st.error(f"Video is larger than {MAX_UPLOAD_MB} MB.")
#             st.stop()

#         st.video(uploaded)

#         if st.button("Analyze video", type="primary"):
#             temp_path = None

#             try:
#                 suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

#                 with tempfile.NamedTemporaryFile(
#                     delete=False,
#                     suffix=suffix,
#                 ) as temp:
#                     temp.write(uploaded.getvalue())
#                     temp_path = temp.name

#                 with st.spinner("Analyzing sampled video frames..."):
#                     result = predict_video(
#                         temp_path,
#                         frame_limit=max_frames,
#                     )

#                 display_result(result)

#                 st.write(
#                     f'**Frames analyzed:** {result["frames"]} '
#                     f'out of {result["total_frames"]} total frames'
#                 )

#                 frame_data = pd.DataFrame(
#                     {
#                         "Frame": result["frame_numbers"],
#                         "Real score (%)": [
#                             score * 100
#                             for score in result["frame_scores"]
#                         ],
#                     }
#                 ).set_index("Frame")

#                 st.subheader("Frame-by-frame analysis")
#                 st.line_chart(frame_data, y="Real score (%)")

#                 st.caption(
#                     "The video prediction aggregates sampled frame scores. "
#                     "This model does not directly analyze temporal motion, "
#                     "and a frame-level score can be unreliable on its own."
#                 )

#             except Exception as exc:
#                 st.error(f"Video analysis failed: {exc}")

#             finally:
#                 if temp_path and os.path.exists(temp_path):
#                     try:
#                         os.remove(temp_path)
#                     except OSError:
#                         pass


# # ---------------- FOOTER ----------------

# st.divider()
# st.caption(
#     "Research prototype only. Results may be incorrect, especially for "
#     "unseen datasets, low-quality media, compression, or unusual lighting. "
#     "Do not use this output alone to make important decisions."
# )


import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Deepfake AI | Media Scanner",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "deepfake_model.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.50
MAX_FRAMES = 24
MAX_UPLOAD_MB = 200


# ---------------- GRAPHICAL DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #172554 0%, transparent 35%),
        linear-gradient(135deg, #070b18 0%, #0c1224 55%, #11132a 100%);
    color: #edf2ff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #0a1020;
    border-right: 1px solid #24304a;
}

[data-testid="stSidebar"] * {
    color: #dce7ff;
}

.hero {
    background: linear-gradient(120deg, #142449, #171a39 65%, #252052);
    border: 1px solid #33466f;
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 45px rgba(0,0,0,.18);
}

.hero-tag {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: 35px;
    font-weight: 800;
    margin: 12px 0 8px 0;
}

.hero p {
    color: #b7c5e4;
    font-size: 15px;
    margin-bottom: 0;
}

.glass-card {
    background: linear-gradient(145deg, #151e33, #101728);
    border: 1px solid #2b3854;
    border-radius: 18px;
    padding: 22px;
    min-height: 130px;
    margin-bottom: 14px;
}

.metric-label {
    color: #a7b5d2;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #7dd3fc;
    font-size: 12px;
    margin-top: 5px;
}

.section-heading {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    margin: 8px 0 14px 0;
}

.upload-card {
    background: #111a2e;
    border: 1px dashed #4b6592;
    border-radius: 20px;
    padding: 22px;
    margin: 10px 0 20px 0;
}

.status-pill {
    display: inline-block;
    border-radius: 30px;
    padding: 8px 15px;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 1px;
    background: #153a38;
    border: 1px solid #27766b;
    color: #6ee7d8;
}

.result-real {
    background: linear-gradient(135deg, #103a35, #142b36);
    border: 1px solid #2c8275;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}

.result-fake {
    background: linear-gradient(135deg, #421e38, #261d3e);
    border: 1px solid #9a477d;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
}

.score {
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
    color: #ffffff;
}

.score-label {
    color: #c0cce5;
    font-size: 13px;
    font-weight: 600;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid #5288ff;
    color: white;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    font-weight: 700;
    min-height: 45px;
    transition: 0.2s;
}

div.stButton > button:hover {
    border-color: #93c5fd;
    color: white;
    box-shadow: 0 0 18px rgba(59,130,246,.3);
}

div[data-testid="stFileUploader"] {
    background: #111a2e;
    border-radius: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: #293650;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.sidebar.markdown("## 🔐 DEEPFAKE AI")
st.sidebar.caption("MEDIA AUTHENTICITY PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Image Scanner", "Video Scanner", "About"],
)

st.sidebar.divider()
st.sidebar.markdown("### 🟢 System status")
st.sidebar.caption("Interface ready for analysis")
st.sidebar.caption("Model: MobileNetV2-based classifier")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Put deepfake_model.keras "
        "beside app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# ---------------- PREDICTION FUNCTIONS ----------------

def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def predict_image(image):
    prediction = model.predict(preprocess_image(image), verbose=0)

    if prediction.size != 1:
        raise ValueError(
            "This app expects a binary classifier with one sigmoid output."
        )

    real_score = float(prediction.reshape(-1)[0])

    if not np.isfinite(real_score) or not 0 <= real_score <= 1:
        raise ValueError("Invalid model output. Check your model architecture.")

    return make_result(real_score)


def make_result(real_score):
    fake_score = 1.0 - real_score

    return {
        "label": "REAL" if real_score >= THRESHOLD else "FAKE",
        "real_score": real_score,
        "fake_score": fake_score,
        "winning_score": max(real_score, fake_score),
    }


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open this video file.")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("This video has no readable frames.")

        positions = np.linspace(
            0,
            total_frames - 1,
            min(MAX_FRAMES, total_frames),
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

                arr = np.expand_dims(
                    frame.astype(np.float32), axis=0
                )

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
                text=f"Analyzing frame {index + 1}/{len(positions)}",
            )

        progress.empty()

        if not scores:
            raise ValueError("No video frames could be analyzed.")

        # Average frame scores for the video-level estimate
        result = make_result(float(np.mean(scores)))
        result["frames"] = len(scores)
        result["total_frames"] = total_frames
        result["frame_scores"] = scores
        result["frame_numbers"] = frame_numbers

        return result

    finally:
        cap.release()


# ---------------- RESULT COMPONENT ----------------

def show_result(result):
    st.markdown("## 🧠 Analysis Result")

    if result["label"] == "REAL":
        st.markdown(
            '<div class="result-real">'
            '<div class="status-pill">✓ LIKELY REAL</div>'
            '<div class="score">'
            f'{result["real_score"] * 100:.1f}%'
            '</div>'
            '<div class="score-label">REAL CLASS SCORE</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-fake">'
            '<div class="status-pill">⚠ POSSIBLE DEEPFAKE</div>'
            '<div class="score">'
            f'{result["fake_score"] * 100:.1f}%'
            '</div>'
            '<div class="score-label">FAKE CLASS SCORE</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="glass-card">'
            '<div class="metric-label">🟢 REAL SCORE</div>'
            f'<div class="metric-value">{result["real_score"] * 100:.2f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.progress(result["real_score"])

    with col2:
        st.markdown(
            '<div class="glass-card">'
            '<div class="metric-label">🟣 FAKE SCORE</div>'
            f'<div class="metric-value">{result["fake_score"] * 100:.2f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.progress(result["fake_score"])

    chart_data = pd.DataFrame({
        "Class": ["Real", "Fake"],
        "Score (%)": [
            result["real_score"] * 100,
            result["fake_score"] * 100,
        ],
    }).set_index("Class")

    st.markdown('<div class="section-heading">📊 Score Comparison</div>',
                unsafe_allow_html=True)
    st.bar_chart(chart_data, y="Score (%)")

    st.caption(
        "These are model scores, not guaranteed probabilities of authenticity. "
        "The classification threshold is fixed at 50%."
    )


def show_upload_metrics(analysis_time, frames=1):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="glass-card">'
            '<div class="metric-label">⚡ ANALYSIS TIME</div>'
            f'<div class="metric-value">{analysis_time:.2f}s</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="glass-card">'
            '<div class="metric-label">🖼 FRAMES ANALYZED</div>'
            f'<div class="metric-value">{frames}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="glass-card">'
            '<div class="metric-label">🧠 MODEL</div>'
            '<div class="metric-value">AI</div>'
            '<div class="metric-note">Deep learning classifier</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------- DASHBOARD ----------------

if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">● AI SYSTEM ONLINE</div>
        <h1>Media Authenticity Scanner</h1>
        <p>
            Inspect digital media for signs of manipulation with a
            deep-learning image classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">🖼 IMAGE ANALYSIS</div>
            <div class="metric-value">Image AI</div>
            <div class="metric-note">Analyze one image at a time</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">🎬 VIDEO ANALYSIS</div>
            <div class="metric-value">Frame AI</div>
            <div class="metric-note">Inspect sampled video frames</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">📈 VISUAL REPORT</div>
            <div class="metric-value">Live Scores</div>
            <div class="metric-note">Compare real and fake scores</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🚀 Start an analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-card">
        <h3>Choose your media scanner</h3>
        <p>
            Use the navigation menu on the left to open Image Scanner
            or Video Scanner.
        </p>
        <p>Supported images: JPG, JPEG, PNG, WEBP</p>
        <p>Supported videos: MP4, AVI, MOV, MKV</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "For best results, use clear images or videos and test the model "
        "with data that was not used during training."
    )


# ---------------- IMAGE SCANNER ----------------

elif page == "Image Scanner":
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">IMAGE INTELLIGENCE</div>
        <h1>🖼 Image Scanner</h1>
        <p>Upload an image and inspect the model's real/fake scores.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
            st.stop()

        try:
            image = Image.open(uploaded)
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")

            left, right = st.columns([1.1, 1])

            with left:
                st.image(image, caption="Uploaded media", use_container_width=True)

            with right:
                st.markdown(
                    '<div class="glass-card">'
                    '<div class="metric-label">FILE INFORMATION</div>'
                    f'<div class="metric-value">{image.width} × {image.height}</div>'
                    '<div class="metric-note">RGB image prepared for analysis</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if st.button("🔍 Analyze Image", type="primary",
                             use_container_width=True):
                    start = time.perf_counter()

                    try:
                        with st.spinner("AI is inspecting the image..."):
                            result = predict_image(image)

                        elapsed = time.perf_counter() - start
                        st.session_state["image_result"] = result
                        st.session_state["image_time"] = elapsed

                    except Exception as exc:
                        st.error(f"Image analysis failed: {exc}")

            if "image_result" in st.session_state:
                st.divider()
                show_result(st.session_state["image_result"])
                show_upload_metrics(
                    st.session_state.get("image_time", 0),
                    frames=1,
                )

        except (UnidentifiedImageError, OSError):
            st.error("The uploaded image is invalid or corrupted.")


# ---------------- VIDEO SCANNER ----------------

elif page == "Video Scanner":
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">VIDEO INTELLIGENCE</div>
        <h1>🎬 Video Scanner</h1>
        <p>Analyze sampled video frames and visualize changes in model scores.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
            st.stop()

        st.video(uploaded)

        if st.button("🎬 Analyze Video", type="primary",
                     use_container_width=True):
            temp_path = None
            start = time.perf_counter()

            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp:
                    temp.write(uploaded.getvalue())
                    temp_path = temp.name

                with st.spinner("Scanning video frames..."):
                    result = predict_video(temp_path)

                elapsed = time.perf_counter() - start
                st.session_state["video_result"] = result
                st.session_state["video_time"] = elapsed

            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        if "video_result" in st.session_state:
            result = st.session_state["video_result"]

            st.divider()
            show_result(result)

            show_upload_metrics(
                st.session_state.get("video_time", 0),
                frames=result["frames"],
            )

            st.markdown(
                '<div class="section-heading">📉 Frame-by-Frame Analysis</div>',
                unsafe_allow_html=True,
            )

            frame_data = pd.DataFrame({
                "Frame number": result["frame_numbers"],
                "Real score (%)": [
                    score * 100 for score in result["frame_scores"]
                ],
            }).set_index("Frame number")

            st.line_chart(frame_data, y="Real score (%)")

            st.caption(
                "Large score variations can indicate inconsistent predictions, "
                "but they do not independently prove that a video is fake."
            )


# ---------------- ABOUT ----------------

elif page == "About":
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PROJECT INFORMATION</div>
        <h1>About Deepfake AI</h1>
        <p>A research prototype for exploring media authenticity classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🧠 How it works</h3>
        1. Upload an image or video.<br><br>
        2. Prepare the image or sample video frames.<br><br>
        3. Run the trained deep learning model.<br><br>
        4. Display real/fake scores and visual comparisons.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "This prototype does not directly analyze temporal motion, "
        "verify the source of media, or guarantee authenticity. "
        "A prediction should not be treated as definitive evidence."
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#8796b5; padding:10px;">
    🔐 DEEPFAKE AI · MEDIA AUTHENTICITY SCANNER
    <br><br>
    <small>AI predictions can be incorrect. Use responsibly.</small>
</div>
""", unsafe_allow_html=True)
