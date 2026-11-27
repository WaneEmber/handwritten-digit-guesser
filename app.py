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
    return (
        joblib.load("knn_model.joblib"),
        joblib.load("svm_model.joblib")
    )


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

st.caption("Dataset accuracy does not guarantee photo recognition accuracy.")


def preprocess_image(image):

    # 1. Correct orientation and grayscale
    image = ImageOps.exif_transpose(image).convert("L")
    gray = np.asarray(image, dtype=np.uint8)

    h, w = gray.shape

    # 2. Reduce small pencil/paper noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # 3. Adaptive threshold
    min_side = min(h, w)

    block_size = min(51, min_side - 1)

    if block_size % 2 == 0:
        block_size -= 1

    block_size = max(3, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7
    )

    # 4. Remove tiny components
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Remove tiny background noise
        if area < max(8, int(h * w * 0.00003)):
            continue

        # Ignore components touching image borders
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue

        components.append(
            (area, i, x, y, cw, ch)
        )

    if not components:
        raise ValueError(
            "Could not detect handwriting. Please try another image."
        )

    # 5. Select the largest handwriting component
    components.sort(
        key=lambda item: item[0],
        reverse=True
    )

    largest = components[0]

    _, main_label, x, y, cw, ch = largest

    # Keep the main connected handwriting component
    main_mask = (
        labels == main_label
    ).astype(np.uint8) * 255

    # 6. Crop tightly around actual handwriting
    crop = main_mask[
        y:y + ch,
        x:x + cw
    ]

    # 7. Add padding
    padding = max(
        2,
        int(max(cw, ch) * 0.18)
    )

    padded_w = cw + padding * 2
    padded_h = ch + padding * 2

    side = max(
        padded_w,
        padded_h
    )

    square = np.zeros(
        (side, side),
        dtype=np.uint8
    )

    start_x = (side - cw) // 2
    start_y = (side - ch) // 2

    square[
        start_y:start_y + ch,
        start_x:start_x + cw
    ] = crop

    # 8. Resize to 8x8
    small = cv2.resize(
        square,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # 9. Normalize brightness
    max_value = float(small.max())

    if max_value > 0:

        small = (
            small.astype(np.float32)
            / max_value
            * 16.0
        )

    small = np.clip(
        small,
        0,
        16
    )

    model_input = small.reshape(
        1, 64
    ).astype(np.float32)

    # 10. Preview
    preview_array = np.rint(
        small / 16 * 255
    ).astype(np.uint8)

    preview = Image.fromarray(
        preview_array
    )

    crop_preview = Image.fromarray(
        crop
    )

    bbox = (
        x,
        y,
        cw,
        ch
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

        st.image(
            original,
            use_container_width=True
        )

        st.caption(
            f"Detected area: x={bbox[0]}, y={bbox[1]}, "
            f"width={bbox[2]}, height={bbox[3]}"
        )

        st.subheader("Detected Handwriting")

        st.image(
            crop_preview,
            use_container_width=True
        )

        st.subheader("Processed 8×8 Image")

        st.image(
            preview.resize(
                (320, 320),
                Image.Resampling.NEAREST
            ),
            caption="Shared input for both models"
        )

        # Same input for KNN and SVM
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

        knn_confidence = (
            float(knn_probs[knn_index]) * 100
        )

        svm_confidence = (
            float(svm_probs[svm_index]) * 100
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
                "Models disagree. The image may be difficult to classify."
            )

        st.subheader("8×8 Pixel Values (0–16)")

        values = np.rint(
            model_input.reshape(8, 8)
        ).astype(int)

        st.dataframe(
            values,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Confidence is an estimated probability, not measured accuracy."
        )

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )


st.markdown("---")

st.caption(
    "Course project: handwritten digit classification using KNN and SVM."
)
