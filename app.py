import streamlit as st
import joblib
import json
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
c1, c2 = st.columns(2)
c1.metric("KNN Accuracy", f"{metrics['results']['knn']['accuracy'] * 100:.2f}%")
c2.metric("SVM Accuracy", f"{metrics['results']['svm']['accuracy'] * 100:.2f}%")
st.caption("Held-out dataset scores are not guarantees for uploaded photographs.")


def preprocess_image(image):
    """One shared input: adaptive threshold -> largest plausible component -> thicken -> 8x8."""
    image = ImageOps.exif_transpose(image).convert("L")
    gray = np.asarray(image, dtype=np.uint8)
    h, w = gray.shape
    min_side = min(h, w)
    block = max(31, min(61, (min_side // 8) | 1))
    if block >= min_side:
        block = max(3, (min_side - 1) | 1)
    c_value = 9 if min_side > 500 else 5

    # Adaptive threshold keeps pencil strokes despite uneven paper illumination.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, c_value
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label in range(1, count):
        x, y, cw, ch, area = map(int, stats[label])
        ratio = cw / max(ch, 1)
        # Ignore frame/edge noise and tiny specks; prefer digit-like components.
        if (x > 0 and y > 0 and x + cw < w and y + ch < h
                and area >= 30 and cw >= 5 and ch >= 8
                and 0.12 <= ratio <= 1.25):
            candidates.append((area, label, x, y, cw, ch))
    if not candidates:
        raise ValueError("Could not isolate a digit. Try a closer, clearer photo with the digit away from the edges.")

    area, label, x, y, cw, ch = max(candidates, key=lambda item: item[0])
    component = (labels[y:y + ch, x:x + cw] == label).astype(np.uint8) * 255

    # Thicken thin pencil strokes before shrinking. This preserves loop structure
    # better than enhancing a blurred grayscale crop.
    component = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=2)
    side = max(cw, ch)
    square = np.zeros((side, side), dtype=np.uint8)
    px, py = (side - cw) // 2, (side - ch) // 2
    square[py:py + ch, px:px + cw] = component

    # Empirically tested fit: 7x7 with the remaining pixel at bottom/right.
    small = cv2.resize(square, (7, 7), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((8, 8), dtype=np.uint8)
    canvas[:7, :7] = small
    # Scale to the Digits dataset's 0..16 range. Both models get exactly this array.
    pixel_data = canvas.astype(np.float32) / 255.0 * 16.0
    crop_preview = Image.fromarray(component)
    # Bright display-only preview; the actual shared model input stays 0..16.
    display_canvas = (canvas.astype(np.float32) / max(float(canvas.max()), 1.0) * 255).astype(np.uint8)
    preview = Image.fromarray(display_canvas)
    bbox = (x, y, cw, ch)
    return pixel_data.reshape(1, 64), crop_preview, preview, bbox


uploaded = st.file_uploader("Upload a handwritten digit photo", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        original = Image.open(uploaded)
        model_input, crop_preview, preview, bbox = preprocess_image(original)
        st.subheader("Original Image")
        st.image(original, use_container_width=True)
        st.caption(f"Detected component: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
        st.subheader("Detected Stroke Crop")
        st.image(crop_preview, use_container_width=True)
        st.subheader("Shared Processed 8×8 Input")
        st.image(preview.resize((320, 320), Image.Resampling.NEAREST), caption="Same exact 8×8 pixels are used by KNN and SVM")

        # IMPORTANT: one input array, passed unchanged to both models.
        knn_prediction = int(knn_model.predict(model_input)[0])
        svm_prediction = int(svm_model.predict(model_input)[0])
        knn_probs = knn_model.predict_proba(model_input)[0]
        svm_probs = svm_model.predict_proba(model_input)[0]
        knn_confidence = float(knn_probs[np.where(knn_model.classes_ == knn_prediction)[0][0]]) * 100
        svm_confidence = float(svm_probs[np.where(svm_model.classes_ == svm_prediction)[0][0]]) * 100

        st.subheader("Predictions")
        c1, c2 = st.columns(2)
        c1.metric("KNN Prediction", knn_prediction)
        c1.write(f"Estimated model confidence: {knn_confidence:.2f}%")
        c2.metric("SVM Prediction", svm_prediction)
        c2.write(f"Estimated model confidence: {svm_confidence:.2f}%")
        if knn_prediction == svm_prediction:
            st.success(f"Both models agree: {knn_prediction}")
        else:
            st.warning("Models disagree. Try a closer crop with the digit centered and clear.")

        st.subheader("Shared 8×8 Pixel Values (0–16)")
        values = np.rint(model_input.reshape(8, 8)).astype(int)
        st.dataframe(values, hide_index=True, use_container_width=True)
        st.caption("Confidence is an estimated probability, not measured accuracy. Real-photo performance can differ from held-out dataset results.")
    except Exception as error:
        st.error(f"Image processing failed: {error}")

st.markdown("---")
st.caption("Course project: handwritten digit classification using KNN and SVM.")
