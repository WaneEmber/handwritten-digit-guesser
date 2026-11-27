
import streamlit as st
import joblib
import json
import numpy as np
import cv2

from PIL import Image, ImageOps


# ==========================================
# 1. Streamlit configuration
# ==========================================

st.set_page_config(
    page_title="AI Handwritten Digit Guesser",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 AI Handwritten Digit Guesser")

st.write(
    "Compare KNN and SVM predictions "
    "for a handwritten digit."
)


# ==========================================
# 2. Load models
# ==========================================

@st.cache_resource
def load_models():

    knn = joblib.load("knn_model.joblib")
    svm = joblib.load("svm_model.joblib")

    return knn, svm


@st.cache_data
def load_metrics():

    with open(
        "metrics.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


try:

    knn_model, svm_model = load_models()
    metrics = load_metrics()

except Exception as error:

    st.error(f"Could not load models: {error}")
    st.stop()


# ==========================================
# 3. Model performance
# ==========================================

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
    "Dataset accuracy does not guarantee photo recognition accuracy."
)


# ==========================================
# 4. OLD-STYLE IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    # 1. Correct orientation and grayscale
    image = ImageOps.exif_transpose(image).convert("L")

    gray = np.asarray(
        image,
        dtype=np.uint8
    )

    h, w = gray.shape

    if min(h, w) < 3:
        raise ValueError("Image is too small.")

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # 3. Adaptive Threshold
    min_side = min(h, w)

    block_size = min(
        51,
        min_side if min_side % 2 == 1
        else min_side - 1
    )

    block_size = max(3, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7
    )

    # 4. Connected Components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])

        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        area = int(stats[i, cv2.CC_STAT_AREA])

        # Ignore very small noise
        if area < max(8, int(h * w * 0.00003)):
            continue

        # Ignore components touching image borders
        if (
            x <= 1
            or y <= 1
            or x + cw >= w - 1
            or y + ch >= h - 1
        ):
            continue

        components.append(
            (area, i, x, y, cw, ch)
        )

    if not components:

        raise ValueError(
            "Could not detect handwriting. "
            "Please try another image."
        )

    # 5. Select largest connected component
    components.sort(
        key=lambda item: item[0],
        reverse=True
    )

    largest = components[0]

    _, main_label, x, y, cw, ch = largest

    main_mask = (
        labels == main_label
    ).astype(np.uint8) * 255

    # 6. Crop handwriting
    crop = main_mask[
        y:y + ch,
        x:x + cw
    ]

    # 7. Make a square image with padding
    padding = max(
        2,
        int(max(cw, ch) * 0.18)
    )

    side = max(cw, ch) + padding * 2

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

    # 8. Resize to 6x6
    small = cv2.resize(
        square,
        (6, 6),
        interpolation=cv2.INTER_AREA
    )

    # 9. Put 6x6 image in the center of 8x8
    final_image = np.zeros(
        (8, 8),
        dtype=np.float32
    )

    final_image[1:7, 1:7] = (
        small.astype(np.float32) / 255.0 * 16.0
    )

    final_image = np.clip(
        final_image,
        0,
        16
    )

    # 10. Model input
    model_input = final_image.reshape(
        1,
        64
    ).astype(np.float32)

    # 11. Preview
    preview_array = np.rint(
        final_image / 16.0 * 255
    ).astype(np.uint8)

    preview = Image.fromarray(
        preview_array
    )

    crop_preview = Image.fromarray(
        crop
    )

    binary_preview = Image.fromarray(
        binary
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
        binary_preview,
        bbox
    )


# ==========================================
# 5. Upload image
# ==========================================

uploaded = st.file_uploader(
    "Upload a handwritten digit photo",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# 6. Image processing
# ==========================================

if uploaded is not None:

    try:

        original = Image.open(uploaded)

        (
            model_input,
            crop_preview,
            preview,
            binary_preview,
            bbox
        ) = preprocess_image(original)

        # Original image
        st.subheader("Original Image")

        st.image(
            original,
            use_container_width=True
        )

        st.caption(
            f"Detected area: "
            f"x={bbox[0]}, y={bbox[1]}, "
            f"width={bbox[2]}, height={bbox[3]}"
        )

        # Binary image
        st.subheader("Binary Image")

        st.image(
            binary_preview,
            use_container_width=True
        )

        # Cropped handwriting
        st.subheader("Detected Handwriting")

        st.image(
            crop_preview,
            use_container_width=True
        )

        # 6x6 centered in 8x8
        st.subheader("Processed 8×8 Image")

        st.image(
            preview.resize(
                (320, 320),
                Image.Resampling.NEAREST
            ),
            caption="6×6 image centered inside an 8×8 canvas"
        )

        # ==================================
        # 7. KNN prediction
        # ==================================

        knn_prediction = int(
            knn_model.predict(model_input)[0]
        )

        knn_probs = knn_model.predict_proba(
            model_input
        )[0]

        knn_index = np.where(
            knn_model.classes_ == knn_prediction
        )[0][0]

        knn_confidence = (
            float(knn_probs[knn_index]) * 100
        )

        # ==================================
        # 8. SVM prediction
        # ==================================

        svm_prediction = int(
            svm_model.predict(model_input)[0]
        )

        svm_probs = svm_model.predict_proba(
            model_input
        )[0]

        svm_index = np.where(
            svm_model.classes_ == svm_prediction
        )[0][0]

        svm_confidence = (
            float(svm_probs[svm_index]) * 100
        )

        # ==================================
        # 9. Display predictions
        # ==================================

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
                "The image may be difficult to classify."
            )

        # ==================================
        # 10. Display 8x8 pixel values
        # ==================================

        st.subheader(
            "8×8 Pixel Values (0–16)"
        )

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
            "not measured accuracy."
        )

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )


# ==========================================
# 11. Footer
# ==========================================

st.markdown("---")

st.caption(
    "Course project: handwritten digit classification using KNN and SVM."
)
