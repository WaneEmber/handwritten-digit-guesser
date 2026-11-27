import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="AI Handwritten Digit Guesser",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 AI Handwritten Digit Guesser")
st.write(
    "Upload a handwritten digit image and let two machine learning "
    "models predict the digit."
)

# =========================
# Load trained models
# =========================
@st.cache_resource
def load_models():
    knn = joblib.load("knn_model.joblib")
    logistic = joblib.load("logistic_model.joblib")
    return knn, logistic

knn_model, logistic_model = load_models()

# =========================
# Image preprocessing
# =========================
def preprocess_image(image):
    # Correct image orientation
    image = ImageOps.exif_transpose(image)

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Remove uneven background using Gaussian blur
    background_image = image.filter(
        ImageFilter.GaussianBlur(radius=30)
    )

    background = np.array(
        background_image,
        dtype=np.float32
    )

    original = np.array(
        image,
        dtype=np.float32
    )

    # Extract dark pencil/ink strokes
    difference = np.clip(
        background - original,
        0,
        255
    )

    max_value = difference.max()

    if max_value < 15:
        raise ValueError(
            "No clear digit detected. Please upload a clearer image."
        )

    # Enhance stroke contrast
    enhanced = (
        difference / max_value * 255
    ).astype(np.uint8)

    # Create a mask to remove weak background noise
    threshold = max(
        25,
        int(enhanced.max() * 0.25)
    )

    mask = enhanced > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError(
            "No clear digit detected. Please try another image."
        )

    # Crop around the digit
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = enhanced[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Keep only the detected digit area
    cropped_mask = mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    cropped = np.where(
        cropped_mask,
        cropped,
        0
    ).astype(np.uint8)

    digit = Image.fromarray(cropped)

    # Resize digit to fit inside a 6x6 area
    digit.thumbnail(
        (6, 6),
        Image.Resampling.LANCZOS
    )

    # Place digit in the center of an 8x8 canvas
    canvas = Image.new(
        "L",
        (8, 8),
        0
    )

    x_offset = (8 - digit.width) // 2
    y_offset = (8 - digit.height) // 2

    canvas.paste(
        digit,
        (x_offset, y_offset)
    )

    # Convert pixel values to sklearn Digits scale (0-16)
    processed = (
        np.array(canvas, dtype=np.float64)
        / 255.0
        * 16.0
    )

    processed = processed.reshape(1, 64)

    return processed, canvas


# =========================
# Upload image
# =========================
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        processed_data, processed_image = preprocess_image(image)

        # Display uploaded and processed images
        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Original uploaded image",
            use_container_width=True
        )

        st.subheader("Processed 8×8 Image")

        st.image(
            processed_image.resize(
                (320, 320),
                Image.Resampling.NEAREST
            ),
            caption="8×8 grayscale image used by the models",
            use_container_width=False
        )

        # =========================
        # Model predictions
        # =========================
        knn_prediction = knn_model.predict(
            processed_data
        )[0]

        logistic_prediction = logistic_model.predict(
            processed_data
        )[0]

        # Probability scores
        knn_probabilities = knn_model.predict_proba(
            processed_data
        )[0]

        logistic_probabilities = logistic_model.predict_proba(
            processed_data
        )[0]

        knn_classes = knn_model.classes_
        logistic_classes = logistic_model.classes_

        knn_confidence = (
            knn_probabilities[
                np.where(knn_classes == knn_prediction)[0][0]
            ] * 100
        )

        logistic_confidence = (
            logistic_probabilities[
                np.where(logistic_classes == logistic_prediction)[0][0]
            ] * 100
        )

        # =========================
        # Display results
        # =========================
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "KNN Prediction",
                str(knn_prediction)
            )

            st.write(
                f"Model confidence: {knn_confidence:.2f}%"
            )

        with col2:
            st.metric(
                "Logistic Regression Prediction",
                str(logistic_prediction)
            )

            st.write(
                f"Model confidence: {logistic_confidence:.2f}%"
            )

        # =========================
        # Agreement result
        # =========================
        if knn_prediction == logistic_prediction:

            st.success(
                f"Both models agree! The predicted digit is "
                f"{knn_prediction}."
            )

        else:

            st.warning(
                "The two models disagree. "
                "Their predictions may be affected by the handwriting "
                "style or image quality."
            )

        # =========================
        # Display pixel values
        # =========================
        st.subheader("8×8 Pixel Values")

        pixel_values = np.round(
            np.array(processed_image) / 255.0 * 16
        ).astype(int)

        st.dataframe(
            pixel_values,
            hide_index=True,
            use_container_width=True
        )

        # =========================
        # Probability disclaimer
        # =========================
        st.caption(
            "Note: Model confidence represents the probability score "
            "estimated by the model. It is not a guarantee of prediction "
            "accuracy."
        )

    except Exception as e:

        st.error(
            f"Error processing image: {e}"
        )

# =========================
# Footer
# =========================
st.markdown("---")

st.caption(
    "AI Handwritten Digit Guesser | "
    "Machine Learning Course Project"
)
