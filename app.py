
import streamlit as st
import numpy as np
import cv2
import joblib
import json

from pathlib import Path
from PIL import Image, ImageDraw


# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Digit Guesser")
st.write("Upload a handwritten digit image and let the AI recognize it.")

BASE_DIR = Path(__file__).resolve().parent


# ======================================
# Load Models
# ======================================

@st.cache_resource
def load_models():

    knn_path = BASE_DIR / "knn_model.joblib"
    svm_path = BASE_DIR / "svm_model.joblib"

    knn = joblib.load(knn_path) if knn_path.exists() else None
    svm = joblib.load(svm_path) if svm_path.exists() else None

    return knn, svm


try:
    knn_model, svm_model = load_models()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ======================================
# Create 8×8 Pixel Grid Preview
# ======================================

def make_pixel_grid(matrix):

    cell_size = 55
    margin = 5
    size = cell_size * 8 + margin * 2

    result = Image.new(
        "RGB",
        (size, size),
        (30, 30, 30)
    )

    draw = ImageDraw.Draw(result)

    for row in range(8):
        for col in range(8):

            value = float(matrix[row, col])

            brightness = int(
                np.clip(value / 16, 0, 1) * 255
            )

            x1 = margin + col * cell_size
            y1 = margin + row * cell_size
            x2 = x1 + cell_size - 2
            y2 = y1 + cell_size - 2

            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(brightness, brightness, brightness),
                outline=(100, 100, 100)
            )

    return result


# ======================================
# Image Preprocessing
# ======================================

def preprocess_image(image):

    # 1. Convert image to grayscale
    rgb = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    # 2. Estimate background brightness
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. Blur the image slightly
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 4. Select threshold direction
    if background_value >= 127:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    # 5. Adaptive thresholding
    block_size = min(51, min(height, width))

    if block_size % 2 == 0:
        block_size -= 1

    block_size = max(3, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        8
    )

    # 6. Detect connected components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        raise ValueError("No handwritten digit was detected.")

    candidates = []

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Ignore components touching image borders
        touches_edge = (
            x == 0 or
            y == 0 or
            x + w >= width or
            y + h >= height
        )

        if not touches_edge and area >= 8:

            candidates.append({
                "id": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": int(area)
            })

    if not candidates:
        raise ValueError(
            "No valid digit region was found. Please check the image."
        )

    # 7. Find the largest component
    largest = max(
        candidates,
        key=lambda item: item["area"]
    )

    largest_area = largest["area"]

    lx = largest["x"]
    ly = largest["y"]
    lw = largest["w"]
    lh = largest["h"]

    # 8. Keep components near the main digit
    nearby_padding = max(
        15,
        int(max(lw, lh) * 0.25)
    )

    expanded_x1 = lx - nearby_padding
    expanded_y1 = ly - nearby_padding
    expanded_x2 = lx + lw + nearby_padding
    expanded_y2 = ly + lh + nearby_padding

    clean = np.zeros_like(binary)

    kept_components = 0

    min_area = max(
        8,
        int(largest_area * 0.01)
    )

    for item in candidates:

        x = item["x"]
        y = item["y"]
        w = item["w"]
        h = item["h"]
        area = item["area"]

        if area < min_area:
            continue

        near_main = not (
            x + w < expanded_x1 or
            x > expanded_x2 or
            y + h < expanded_y1 or
            y > expanded_y2
        )

        if near_main:

            clean[labels == item["id"]] = 255
            kept_components += 1

    if np.count_nonzero(clean) == 0:
        raise ValueError("No valid digit strokes were found.")

    # 9. Find digit boundaries
    points = cv2.findNonZero(clean)

    x, y, box_width, box_height = cv2.boundingRect(points)

    # Prevent the entire background from being treated as a digit
    if (
        box_width > width * 0.85 or
        box_height > height * 0.85
    ):
        raise ValueError(
            "The detected region includes too much background. "
            "Please use an image with a cleaner background."
        )

    # 10. Add a small margin
    padding = max(
        3,
        int(max(box_width, box_height) * 0.08)
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop_mask = clean[y1:y2, x1:x2]

    crop_height, crop_width = crop_mask.shape

    # ==================================
    # Preserve Original Grayscale Strokes
    # ==================================

    # 11. Estimate local background
    sigma = max(
        5,
        min(height, width) * 0.025
    )

    background_surface = cv2.GaussianBlur(
        gray.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    if background_value >= 127:

        ink_strength = (
            background_surface - gray.astype(np.float32)
        )

    else:

        ink_strength = (
            gray.astype(np.float32) - background_surface
        )

    ink_strength = np.clip(
        ink_strength,
        0,
        255
    )

    # 12. Keep grayscale stroke information
    crop_gray = ink_strength[y1:y2, x1:x2]

    crop_gray = cv2.bitwise_and(
        crop_gray.astype(np.float32),
        crop_gray.astype(np.float32),
        mask=crop_mask
    )

    crop_gray = np.clip(
        crop_gray,
        0,
        255
    ).astype(np.float32)

    # 13. Normalize stroke intensity
    positive_pixels = crop_gray[crop_gray > 0]

    if len(positive_pixels) == 0:
        raise ValueError("No valid grayscale strokes were extracted.")

    intensity_reference = float(
        np.percentile(positive_pixels, 95)
    )

    if intensity_reference < 1:
        raise ValueError("Stroke brightness is too low.")

    crop_gray = (
        crop_gray / intensity_reference * 255.0
    )

    crop_gray = np.clip(
        crop_gray,
        0,
        255
    ).astype(np.uint8)

    # ==================================
    # Resize to 8×8
    # ==================================

    # 14. Maintain aspect ratio and fit within 7×7
    target_size = 7

    scale = target_size / max(
        crop_width,
        crop_height
    )

    new_width = max(
        1,
        min(7, int(round(crop_width * scale)))
    )

    new_height = max(
        1,
        min(7, int(round(crop_height * scale)))
    )

    resized = cv2.resize(
        crop_gray,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # 15. Place digit in the center of an 8×8 canvas
    small = np.zeros(
        (8, 8),
        dtype=np.uint8
    )

    start_x = (8 - new_width) // 2
    start_y = (8 - new_height) // 2

    small[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    # 16. Convert pixel values to the 0–16 range
    processed = (
        small.astype(np.float32) / 255.0 * 16.0
    )

    processed = np.clip(
        processed,
        0,
        16
    ).astype(np.float32)

    model_input = processed.reshape(1, 64)

    # ==================================
    # Generate Previews
    # ==================================

    binary_preview = Image.fromarray(clean)

    crop_preview = Image.fromarray(crop_gray)

    resized_preview = Image.fromarray(small).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    processed_display = np.uint8(
        np.round(processed / 16.0 * 255.0)
    )

    processed_preview = Image.fromarray(
        processed_display
    ).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    grid_preview = make_pixel_grid(processed)

    # ==================================
    # Debug Information
    # ==================================

    debug = {
        "Original image size": f"{width} × {height}",
        "Background brightness": round(background_value, 2),
        "Digit bounding box": f"{box_width} × {box_height}",
        "Cropped image size": f"{crop_width} × {crop_height}",
        "Resized digit size": f"{new_width} × {new_height}",
        "Retained connected components": kept_components,
        "Stroke intensity reference": round(intensity_reference, 2),
        "Non-zero 8×8 pixels": int(np.count_nonzero(processed)),
        "Maximum 8×8 pixel value": round(float(processed.max()), 2)
    }

    return (
        model_input,
        binary_preview,
        crop_preview,
        resized_preview,
        processed_preview,
        grid_preview,
        processed,
        debug
    )


# ======================================
# Prediction
# ======================================

def predict_digit(model, model_input):

    prediction = model.predict(model_input)[0]

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(model_input)[0]

        confidence = float(
            np.max(probabilities)
        ) * 100

    return prediction, confidence


# ======================================
# Main Application
# ======================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, width=300)

        (
            model_input,
            binary_preview,
            crop_preview,
            resized_preview,
            processed_preview,
            grid_preview,
            processed,
            debug
        ) = preprocess_image(image)

        # ==================================
        # Preprocessing Results
        # ==================================

        st.divider()
        st.subheader("Image Preprocessing")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Binary Image (Digit Detection)**")
            st.image(binary_preview, width=300)

        with col2:

            st.markdown("**Detected Handwriting (Grayscale)**")
            st.image(crop_preview, width=300)

        st.divider()

        st.markdown("### Resized Digit (8×8 Enlarged Preview)")
        st.image(resized_preview, width=400)

        st.markdown("### Processed 8×8 Image")
        st.image(processed_preview, width=400)

        st.markdown("### 8×8 Pixel Grid")
        st.image(grid_preview, width=460)

        # ==================================
        # Pixel Matrix
        # ==================================

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            np.rint(processed).astype(int),
            use_container_width=True
        )

        # ==================================
        # Debug Information
        # ==================================

        with st.expander("Preprocessing Debug Information"):

            for key, value in debug.items():

                st.write(f"**{key}:** {value}")

        # ==================================
        # Recognition Results
        # ==================================

        st.divider()
        st.subheader("Recognition Results")

        col1, col2 = st.columns(2)

        if knn_model is not None:

            with col1:

                st.markdown("### KNN")

                prediction, confidence = predict_digit(
                    knn_model,
                    model_input
                )

                st.metric(
                    "Predicted Digit",
                    str(prediction)
                )

                if confidence is not None:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

        if svm_model is not None:

            with col2:

                st.markdown("### SVM")

                prediction, confidence = predict_digit(
                    svm_model,
                    model_input
                )

                st.metric(
                    "Predicted Digit",
                    str(prediction)
                )

                if confidence is not None:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

    except Exception as e:

        st.error(f"Image processing failed: {e}")

else:

    st.info("Please upload a handwritten digit image to get started.")


# ======================================
# Model Evaluation Information
# ======================================

metrics_path = BASE_DIR / "metrics.json"

if metrics_path.exists():

    with st.expander("Model Evaluation Metrics"):

        try:

            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as f:

                st.json(json.load(f))

        except Exception as e:

            st.warning(f"Could not load metrics.json: {e}")
