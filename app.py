import streamlit as st
import joblib, json
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(page_title="AI Handwritten Digit Guesser", page_icon="🔢", layout="centered")
st.title("🔢 AI Handwritten Digit Guesser")
st.write("Compare KNN and SVM predictions for a handwritten digit.")

@st.cache_resource
def load_models():
    return joblib.load("knn_model.joblib"), joblib.load("svm_model.joblib")

@st.cache_data
def load_metrics():
    with open("metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

knn_model, svm_model = load_models()
metrics = load_metrics()

st.subheader("Model Test-Set Performance")
m1, m2 = st.columns(2)
m1.metric("KNN Accuracy", f"{metrics['results']['knn']['accuracy']*100:.2f}%")
m2.metric("SVM Accuracy", f"{metrics['results']['svm']['accuracy']*100:.2f}%")
st.caption("Held-out dataset scores; uploaded-photo performance may differ.")

def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    sigma = max(4, min(image.size) / 30)
    background = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    difference = cv2.subtract(background, arr)
    positive = difference[difference > 0]
    if positive.size == 0:
        raise ValueError("No dark strokes detected. Try a clearer photo.")

    threshold = max(5, int(np.percentile(positive, 75)))
    binary = (difference > threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    H, W = binary.shape
    candidates = []
    for component_id, stat in enumerate(stats[1:], start=1):
        x, y, w, h, area = [int(v) for v in stat]
        if area >= 30 and x > 0 and y > 0 and x+w < W and y+h < H:
            candidates.append((component_id, x, y, w, h, area))

    if candidates:
        component_id, x, y, w, h, _ = max(candidates, key=lambda item: item[5])
        component = labels[y:y+h, x:x+w] == component_id
        crop = np.where(component, difference[y:y+h, x:x+w], 0).astype(np.uint8)
    else:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            raise ValueError("Could not isolate digit. Try cropping closer.")
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = np.where(binary[y0:y1+1, x0:x1+1], difference[y0:y1+1, x0:x1+1], 0).astype(np.uint8)

    if crop.max() == 0:
        raise ValueError("Detected region is too faint. Try a clearer photo.")
    crop = (crop.astype(np.float32) / float(crop.max()) * 255).astype(np.uint8)

    ch, cw = crop.shape
    side = max(ch, cw)
    square = Image.new("L", (side, side), 0)
    square.paste(Image.fromarray(crop), ((side-cw)//2, (side-ch)//2))
    square.thumbnail((6, 6), Image.Resampling.LANCZOS)
    base_canvas = Image.new("L", (8, 8), 0)
    base_canvas.paste(square, ((8-square.width)//2, (8-square.height)//2))

    # Keep softer contrast for KNN; normalize resized intensity for SVM.
    knn_canvas = base_canvas.copy()
    knn_data = np.asarray(knn_canvas, dtype=np.float32).reshape(1, 64) / 255.0 * 16.0
    base = np.asarray(base_canvas, dtype=np.float32)
    svm_canvas = Image.fromarray(np.clip(base / max(float(base.max()), 1) * 255, 0, 255).astype(np.uint8))
    svm_data = np.asarray(svm_canvas, dtype=np.float32).reshape(1, 64) / 255.0 * 16.0
    return knn_data, svm_data, knn_canvas, svm_canvas

uploaded = st.file_uploader("Upload a digit photo (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded:
    try:
        original = Image.open(uploaded)
        knn_input, svm_input, knn_canvas, svm_canvas = preprocess_image(original)
        st.subheader("Uploaded Photo")
        st.image(original, caption="Original image", use_container_width=True)
        st.subheader("Processed Model Inputs")
        c1, c2 = st.columns(2)
        c1.image(knn_canvas.resize((320, 320), Image.Resampling.NEAREST), caption="KNN input")
        c2.image(svm_canvas.resize((320, 320), Image.Resampling.NEAREST), caption="SVM input")

        kp = int(knn_model.predict(knn_input)[0])
        sp = int(svm_model.predict(svm_input)[0])
        kprob = knn_model.predict_proba(knn_input)[0]
        sprob = svm_model.predict_proba(svm_input)[0]
        kconf = float(kprob[np.where(knn_model.classes_ == kp)[0][0]]) * 100
        sconf = float(sprob[np.where(svm_model.classes_ == sp)[0][0]]) * 100
        st.subheader("Predictions")
        c1, c2 = st.columns(2)
        c1.metric("KNN Prediction", kp)
        c1.write(f"Estimated model confidence: {kconf:.2f}%")
        c2.metric("SVM Prediction", sp)
        c2.write(f"Estimated model confidence: {sconf:.2f}%")
        if kp == sp:
            st.success(f"Both models agree: {kp}")
        else:
            st.warning("Models disagree. Try a closer crop with the digit centered and clear.")

        st.subheader("8×8 Pixel Values (0–16)")
        t1, t2 = st.columns(2)
        t1.write("KNN input")
        t1.dataframe(np.rint(np.asarray(knn_canvas, dtype=float)/255*16).astype(int), hide_index=True)
        t2.write("SVM input")
        t2.dataframe(np.rint(np.asarray(svm_canvas, dtype=float)/255*16).astype(int), hide_index=True)
        st.caption("Confidence is the model-estimated probability, not measured accuracy. Photo inputs may differ from the training dataset.")
    except Exception as err:
        st.error(f"Image processing failed: {err}")

st.markdown("---")
st.caption("Course project: handwritten digit classification using KNN and SVM.")
