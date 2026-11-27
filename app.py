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
st.caption("Scores are measured on a held-out dataset; uploaded-photo accuracy can differ.")


def preprocess_image(image):
    """Extract dark strokes using local-background subtraction, then center on 8x8."""
    image = ImageOps.exif_transpose(image).convert("L")
    gray = np.asarray(image, dtype=np.uint8)
    h_img, w_img = gray.shape

    # Scale the blur with image size to estimate uneven paper illumination.
    sigma = max(4.0, min(h_img, w_img) / 30.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    difference = cv2.subtract(background, gray)
    positive = difference[difference > 0]
    if positive.size == 0:
        raise ValueError("No dark strokes found. Try a clearer image.")

    # Threshold weak texture; keep pencil strokes and their connected shape.
    threshold = max(5, int(np.percentile(positive, 75)))
    binary = (difference > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    # Exclude edge-touching components and tiny noise. The largest remaining
    # plausible component is usually the handwritten digit.
    candidates = []
    for label in range(1, n_labels):
        x, y, w, h, area = map(int, stats[label])
        if area >= 30 and x > 0 and y > 0 and x + w < w_img and y + h < h_img:
            if w >= 5 and h >= 8 and area < (h_img * w_img * 0.08):
                candidates.append((label, x, y, w, h, area))

    if not candidates:
        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            raise ValueError("Could not isolate the digit. Try a closer crop.")
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = np.where(binary[y0:y1+1, x0:x1+1], difference[y0:y1+1, x0:x1+1], 0).astype(np.uint8)
        bbox = (int(x0), int(y0), int(x1-x0+1), int(y1-y0+1))
    else:
        label, x, y, w, h, _ = max(candidates, key=lambda item: item[5])
        component = labels[y:y+h, x:x+w] == label
        crop = np.where(component, difference[y:y+h, x:x+w], 0).astype(np.uint8)
        bbox = (x, y, w, h)

    if crop.max() == 0:
        raise ValueError("Detected strokes are too faint. Try a clearer photo.")

    # Normalize extracted strokes before resampling. This avoids pencil gray
    # values collapsing to nearly-zero pixels on the 8x8 input.
    crop = (crop.astype(np.float32) / float(crop.max()) * 255).astype(np.uint8)
    ch, cw = crop.shape
    side = max(ch, cw)
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side-ch)//2:(side-ch)//2+ch, (side-cw)//2:(side-cw)//2+cw] = crop

    # Fit within 6x6, leaving a 1-pixel margin in the 8x8 canvas.
    small = cv2.resize(square, (6, 6), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((8, 8), dtype=np.uint8)
    canvas[1:7, 1:7] = small

    # KNN keeps the normalized soft pixel intensities; SVM receives the same
    # shape with intensity scaled to the full 0..16 range.
    knn_data = canvas.astype(np.float32).reshape(1, 64) / 255.0 * 16.0
    svm_data = canvas.astype(np.float32).reshape(1, 64) / max(float(canvas.max()), 1.0) * 16.0
    cropped_preview = Image.fromarray(crop)
    knn_preview = Image.fromarray(canvas)
    svm_preview = Image.fromarray(np.clip(canvas.astype(float) / max(float(canvas.max()), 1.0) * 255, 0, 255).astype(np.uint8))
    return knn_data, svm_data, cropped_preview, knn_preview, svm_preview, bbox

uploaded = st.file_uploader("Upload a digit photo (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded:
    try:
        original = Image.open(uploaded)
        knn_input, svm_input, crop_preview, knn_canvas, svm_canvas, bbox = preprocess_image(original)
        st.subheader("Uploaded Photo")
        st.image(original, caption="Original uploaded image", use_container_width=True)
        st.caption(f"Detected region: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
        st.subheader("Detected Digit Crop")
        st.image(crop_preview, caption="Extracted stroke region", use_container_width=True)
        st.subheader("Processed 8×8 Inputs")
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
        c1, c2 = st.columns(2)
        c1.write("KNN input")
        c1.dataframe(np.rint(np.asarray(knn_canvas, dtype=float) / 255 * 16).astype(int), hide_index=True)
        c2.write("SVM input")
        c2.dataframe(np.rint(np.asarray(svm_canvas, dtype=float) / 255 * 16).astype(int), hide_index=True)
        st.caption("Confidence is a model-estimated probability, not measured accuracy. Real-photo performance can differ from the held-out dataset.")
    except Exception as err:
        st.error(f"Image processing failed: {err}")

st.markdown("---")
st.caption("Course project: handwritten digit classification using KNN and SVM.")
