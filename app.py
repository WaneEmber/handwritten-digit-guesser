
import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="🔢",
    layout="centered"
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

    # 1. Correct image orientation
    image = ImageOps.exif_transpose(image)

    # 2. Convert to grayscale
    image = ImageOps.grayscale(image)

    img_array = np.array(image, dtype=np.uint8)

    # 3. Estimate uneven paper background
    background_image = image.filter(
        ImageFilter.GaussianBlur(radius=30)
    )

    background = np.array(
        background_image,
        dtype=np.float32
    )

    original = img_array.astype(np.float32)

    # 4. Extract dark pencil strokes
    difference = np.clip(
        background - original,
        0,
        255
    )

    # 5. Enhance contrast
    max_value = float(difference.max())

    if max_value < 15:
        raise ValueError(
            "No clear digit detected. "
            "Please upload a clearer image."
        )

    enhanced = np.clip(
        difference / max_value * 255,
        0,
        255
    ).astype(np.uint8)

    # 6. Remove weak background noise
    threshold = max(
        25,
        int(enhanced.max() * 0.25)
    )

    mask = enhanced > threshold

    if not np.any(mask):
        raise ValueError(
            "No clear digit detected."
        )

    # 7. Find digit boundaries
    ys, xs = np.where(mask)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 8. Crop digit area
    digit_array = enhanced[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    digit_mask = mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Remove pixels outside the detected digit
    digit_array = np.where(
        digit_mask,
        digit_array,
        0
    ).astype(np.uint8)

    digit = Image.fromarray(digit_array)

    # 9. Resize while preserving aspect ratio
    digit.thumbnail(
        (6, 6),
        Image.Resampling.LANCZOS
    )

    # 10. Place digit in center of 8x8 canvas
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

    # 11. Convert pixels to sklearn digits range 0-16
    processed = (
        np.array(canvas, dtype=np.float64)
        / 255.0
        * 16.0
    )

    return processed.reshape(1, 64), canvas


# =========================
# Main interface
# =========================
st.title("🔢 Handwritten Digit Guesser")

st.write(
    "Upload an image of a handwritten digit from 0 to 9. "
    "Two machine learning models will predict the digit."
)

st.info(
    "For best results, upload a clear image containing "
    "one handwritten digit with a plain background."
)

uploaded_file = st.file_uploader(
    "Upload your handwritten digit",
    type=["png", "jpg", "jpeg"]
)


# =========================
# Image upload and display
# =========================
if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        original_image,
        width=250
    )

    # =========================
    # Recognition
    # =========================
    if st.button(
        "Recognize Digit",
        type="primary"
    ):

        try:

            input_data, processed_image = preprocess_image(
                original_image
            )

            # =========================
            # Predictions
            # =========================
            knn_prediction = int(
                knn_model.predict(input_data)[0]
            )

            logistic_prediction = int(
                logistic_model.predict(input_data)[0]
            )

            # =========================
            # Probability scores
            # =========================
            knn_probabilities = (
                knn_model.predict_proba(input_data)[0]
            )

            logistic_probabilities = (
                logistic_model.predict_proba(input_data)[0]
            )

            knn_score = (
                float(np.max(knn_probabilities))
                * 100
            )

            logistic_score = (
                float(np.max(logistic_probabilities))
                * 100
            )

            # =========================
            # Display results
            # =========================
            st.divider()

            st.subheader("Recognition Results")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "KNN Prediction",
                    knn_prediction
                )

                st.write(
                    f"Model probability score: "
                    f"{knn_score:.2f}%"
                )

            with col2:

                st.metric(
                    "Logistic Regression",
                    logistic_prediction
                )

                st.write(
                    f"Model probability score: "
                    f"{logistic_score:.2f}%"
                )

            # =========================
            # Display processed image
            # =========================
            st.subheader("Processed 8 × 8 Image")

            enlarged_image = processed_image.resize(
                (240, 240),
                Image.Resampling.NEAREST
            )

            st.image(
                enlarged_image,
                caption="Processed grayscale image",
                width=240
            )

            # =========================
            # Display pixel values
            # =========================
            st.subheader("8 × 8 Pixel Values")

            pixel_values = np.round(
                np.array(processed_image)
                / 255.0
                * 16
            ).astype(int)

            st.dataframe(
                pixel_values,
                use_container_width=True
            )

            # =========================
            # Compare models
            # =========================
            if knn_prediction == logistic_prediction:

                st.success(
                    f"Both models predict the digit: "
                    f"{knn_prediction}"
                )

            else:

                st.warning(
                    "The models disagree. "
                    "Please check the processed image."
                )

            # =========================
            # Disclaimer
            # =========================
            st.caption(
                "Probability scores are model outputs, "
                "not guaranteed measures of real-world accuracy."
            )

        except Exception as error:

            st.error(
                f"Recognition failed: {error}"
            )


# =========================
# Footer
# =========================
st.divider()

st.caption(
    "AI Handwritten Digit Guesser | "
    "KNN & Logistic Regression"
)
