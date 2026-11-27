import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="🔢",
    layout="centered"
)


# Load trained models
@st.cache_resource
def load_models():
    knn = joblib.load("knn_model.joblib")
    logistic = joblib.load("logistic_model.joblib")
    return knn, logistic


knn_model, logistic_model = load_models()


# Calculate Otsu threshold
def otsu_threshold(array):
    values = np.asarray(array, dtype=np.uint8)

    histogram = np.bincount(values.ravel(), minlength=256)

    total = values.size
    total_sum = np.dot(np.arange(256), histogram)

    background_weight = 0
    background_sum = 0

    best_variance = -1
    best_threshold = 0

    for threshold in range(256):

        background_weight += histogram[threshold]
        background_sum += threshold * histogram[threshold]

        foreground_weight = total - background_weight

        if background_weight == 0 or foreground_weight == 0:
            continue

        background_mean = background_sum / background_weight

        foreground_mean = (
            total_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


# Preprocess uploaded image
def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = ImageOps.grayscale(image)

    img_array = np.array(image)

    # Detect background using border brightness
    border = np.concatenate([
        img_array[0, :],
        img_array[-1, :],
        img_array[:, 0],
        img_array[:, -1]
    ])

    # White/light background -> invert
    if np.median(border) > 127:
        img_array = 255 - img_array

    # Find digit area
    threshold = max(20, int(img_array.max() * 0.15))

    ys, xs = np.where(img_array > threshold)

    if len(xs) == 0:
        raise ValueError("No clear digit detected.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    digit = Image.fromarray(
        img_array[y_min:y_max + 1, x_min:x_max + 1]
    )

    # Preserve aspect ratio
    digit.thumbnail((7, 7), Image.Resampling.LANCZOS)

    # Center on black canvas
    canvas = Image.new("L", (8, 8), 0)

    x_offset = (8 - digit.width) // 2
    y_offset = (8 - digit.height) // 2

    canvas.paste(digit, (x_offset, y_offset))

    # Convert to 0-16 range
    processed = (
        np.array(canvas, dtype=np.float64)
        / 255.0 * 16.0
    )

    return processed.reshape(1, 64), canvas


# Main interface
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


if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(original_image, width=250)

    if st.button("Recognize Digit", type="primary"):

        try:

            input_data, processed_image = preprocess_image(
                original_image
            )

            # Get predictions
            knn_prediction = int(
                knn_model.predict(input_data)[0]
            )

            logistic_prediction = int(
                logistic_model.predict(input_data)[0]
            )

            # Get model probability scores
            knn_probabilities = (
                knn_model.predict_proba(input_data)[0]
            )

            logistic_probabilities = (
                logistic_model.predict_proba(input_data)[0]
            )

            knn_score = (
                float(np.max(knn_probabilities)) * 100
            )

            logistic_score = (
                float(np.max(logistic_probabilities)) * 100
            )

            # Display results
            st.divider()
            st.subheader("Recognition Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "KNN Prediction",
                    knn_prediction
                )

                st.write(
                    f"Model probability score: {knn_score:.2f}%"
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

            # Display processed image
            st.subheader("Processed 8 × 8 Image")

            enlarged_image = processed_image.resize(
                (240, 240),
                Image.Resampling.NEAREST
            )

            st.image(
                enlarged_image,
                caption="Enhanced image converted to 8 × 8 pixels",
                width=240
            )

            # Display actual pixel values
            st.subheader("8 × 8 Pixel Values")

            pixel_values = np.array(processed_image)

            st.dataframe(
                pixel_values,
                use_container_width=True
            )

            # Compare predictions
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

            st.caption(
                "Probability scores are model outputs, "
                "not guaranteed measures of real-world accuracy."
            )

        except Exception as error:

            st.error(
                f"Recognition failed: {error}"
            )


st.divider()

st.caption(
    "AI Handwritten Digit Guesser | "
    "KNN & Logistic Regression"
)
