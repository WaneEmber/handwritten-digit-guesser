import streamlit as st
import joblib
import json
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(
    page_title="AI Handwritten Digit Guesser",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 AI Handwritten Digit Guesser")
st.write("Compare KNN and SVM predictions for a handwritten digit.")


@st.cache_resource
def load_models():
    knn = joblib.load("knn_model.joblib")
    svm = joblib.load("svm_model.joblib")
    return knn, svm


@st.cache_data
def load_metrics():
    with open("metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


knn_model, svm_model = load_models()
metrics = load_metrics()

st.subheader("Model Test-Set Performance")

c1, c2 = st.columns(2)

c1.metric(
    "KNN Accuracy",
    f"{metrics['results']['knn']['accuracy'] * 100:.2f}%"
)

c2.metric(
    "SVM Accuracy",
    f"{metrics['results']['svm']['accuracy'] * 100:.2f}%"
)

st.caption(
    "Held-out dataset scores are not guarantees "
    "for uploaded photographs."
)


def preprocess_image(image):
    """
    Improved image preprocessing:
    1. Grayscale conversion
    2. Adaptive thresholding
    3. Remove small noise components
    4. Keep all digit strokes
    5. Crop and center the digit
    6. Resize to 8x8
    7. Convert pixels to 0-16
    """

    # Correct orientation and convert to grayscale
    image = ImageOps.exif_transpose(image).convert("L")
    gray = np.asarray(image, dtype=np.uint8)

    h, w = gray.shape
    min_side = min(h, w)

    # Adaptive threshold block size
    block = min(51, min_side - 1)

    if block % 2 == 0:
        block -= 1

    block = max(block, 3)

    # Detect pencil strokes
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        5
    )

    # Remove tiny isolated noise without thickening strokes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    # Keep small disconnected stroke fragments too
    min_area = 3

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    # Find all remaining foreground pixels
    ys, xs = np.where(cleaned > 0)

    if len(xs) == 0:
        raise ValueError(
            "No digit detected. Please upload a clearer image."
        )

    # Bounding box of all strokes, not just the largest component
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    crop = cleaned[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    crop_h, crop_w = crop.shape

    # Add padding so the digit doesn't touch the edge
    padding = max(2, int(max(crop_h, crop_w) * 0.15))

    padded_h = crop_h + padding * 2
    padded_w = crop_w + padding * 2

    side = max(padded_h, padded_w)

    square = np.zeros((side, side), dtype=np.uint8)

    # Center the digit in a square canvas
    offset_y = (side - padded_h) // 2
    offset_x = (side - padded_w) // 2

    square[
        offset_y + padding:
        offset_y + padding + crop_h,
        offset_x + padding:
        offset_x + padding + crop_w
    ] = crop

    # Resize directly to 8x8
    small = cv2.resize(
        square,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # Convert to the Digits dataset pixel scale: 0-16
    pixel_data = small.astype(np.float32) / 255.0 * 16.0

    model_input = pixel_data.reshape(1, 64)

    # Display enlarged processed image
    preview_array = np.rint(
        pixel_data / 16.0 * 255
    ).astype(np.uint8)

    preview = Image.fromarray(preview_array)

    crop_preview = Image.fromarray(crop)

    bbox = (
        x_min,
        y_min,
        crop_w,
        crop_h
    )

    return (
        model_input,
        crop_preview,
        preview,
        bbox
    )


uploaded = st.file_uploader(
    "Upload a handwritten digit photo",
    type=["png", "jpg", "jpeg"]
)


if uploaded is not None:

    try:

        original = Image.open(uploaded)

        (
            model_input,
            crop_preview,
            preview,
            bbox
        ) = preprocess_image(original)

        st.subheader("Original Image")
        st.image(original, use_container_width=True)

        st.caption(
            f"Detected digit area: "
            f"x={bbox[0]}, y={bbox[1]}, "
            f"width={bbox[2]}, height={bbox[3]}"
        )

        st.subheader("Detected Digit Strokes")
        st.image(
            crop_preview,
            use_container_width=True
        )

        st.subheader("Shared Processed 8×8 Input")

        st.image(
            preview.resize(
                (320, 320),
                Image.Resampling.NEAREST
            ),
            caption="Same exact pixels are used by KNN and SVM"
        )

        # Both models receive the exact same input
        knn_prediction = int(
            knn_model.predict(model_input)[0]
        )

        svm_prediction = int(
            svm_model.predict(model_input)[0]
        )

        knn_probs = knn_model.predict_proba(model_input)[0]
        svm_probs = svm_model.predict_proba(model_input)[0]

        knn_index = np.where(
            knn_model.classes_ == knn_prediction
        )[0][0]

        svm_index = np.where(
            svm_model.classes_ == svm_prediction
        )[0][0]

        knn_confidence = float(
            knn_probs[knn_index] * 100
        )

        svm_confidence = float(
            svm_probs[svm_index] * 100
        )

        st.subheader("Predictions")

        c1, c2 = st.columns(2)

        c1.metric(
            "KNN Prediction",
            knn_prediction
        )

        c1.write(
            f"Estimated model confidence: "
            f"{knn_confidence:.2f}%"
        )

        c2.metric(
            "SVM Prediction",
            svm_prediction
        )

        c2.write(
            f"Estimated model confidence: "
            f"{svm_confidence:.2f}%"
        )

        if knn_prediction == svm_prediction:

            st.success(
                f"Both models agree: {knn_prediction}"
            )

        else:

            st.warning(
                "Models disagree. "
                "The uploaded image may be difficult to classify."
            )

        st.subheader("Shared 8×8 Pixel Values (0–16)")

        values = np.rint(
            model_input.reshape(8, 8)
        ).astype(int)

        st.dataframe(
            values,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Confidence is an estimated probability, "
            "not measured accuracy. "
            "Real-photo performance can differ from "
            "held-out dataset results."
        )

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )


st.markdown("---")

st.caption(
    "Course project: handwritten digit classification using KNN and SVM."
)
