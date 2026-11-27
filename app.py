
import streamlit as st
import numpy as np
import cv2
import joblib
import json

from pathlib import Path
from PIL import Image


# ==========================================
# 页面设置
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Digit Guesser")
st.write("上传一张手写数字图片，使用 KNN 和 SVM 模型进行识别。")

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 加载模型
# ==========================================

@st.cache_resource
def load_models():
    knn_path = BASE_DIR / "knn_model.joblib"
    svm_path = BASE_DIR / "svm_model.joblib"

    knn_model = None
    svm_model = None

    if knn_path.exists():
        knn_model = joblib.load(knn_path)

    if svm_path.exists():
        svm_model = joblib.load(svm_path)

    return knn_model, svm_model


try:
    knn_model, svm_model = load_models()
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()


# ==========================================
# 图片预处理
# ==========================================

def preprocess_image(image):
    """
    将上传的手写数字图片转换为 sklearn Digits Dataset
    所使用的 8×8、0-16 像素格式。
    """

    # 1. PIL 转 OpenCV 灰度图
    image_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 2. 降低照片噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. 自适应二值化
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    # 4. 自动判断背景颜色，确保笔画为白色、背景为黑色
    h, w = binary.shape

    border_pixels = np.concatenate([
        binary[0, :],
        binary[-1, :],
        binary[:, 0],
        binary[:, -1]
    ])

    border_mean = np.mean(border_pixels)

    if border_mean > 127:
        binary = 255 - binary

    # 5. 去除小噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    clean = np.zeros_like(binary)

    min_area = max(3, int(h * w * 0.0005))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255

    # 如果没有检测到笔画，使用原始二值图
    if np.count_nonzero(clean) == 0:
        clean = binary.copy()

    # 6. 获取数字整体边界
    points = cv2.findNonZero(clean)

    if points is None:
        raise ValueError("没有检测到有效笔画，请使用背景干净、数字清晰的图片。")

    x, y, box_w, box_h = cv2.boundingRect(points)

    crop = clean[y:y + box_h, x:x + box_w]

    # 7. 保持数字宽高比例，放入正方形画布
    size = max(box_w, box_h)

    padding = max(2, int(size * 0.20))
    canvas_size = size + padding * 2

    square = np.zeros(
        (canvas_size, canvas_size),
        dtype=np.uint8
    )

    offset_x = (canvas_size - box_w) // 2
    offset_y = (canvas_size - box_h) // 2

    square[
        offset_y:offset_y + box_h,
        offset_x:offset_x + box_w
    ] = crop

    # 8. 先放大到 24×24，保留笔画细节
    medium = cv2.resize(
        square,
        (24, 24),
        interpolation=cv2.INTER_NEAREST
    )

    # 9. 轻微加粗笔画
    kernel = np.ones((2, 2), np.uint8)

    medium = cv2.dilate(
        medium,
        kernel,
        iterations=1
    )

    # 10. 缩小到模型需要的 8×8
    small = cv2.resize(
        medium,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # 11. 映射到 0-16
    processed = np.round(
        small.astype(np.float32) / 255.0 * 16.0
    )

    processed = np.clip(processed, 0, 16).astype(np.float32)

    # 12. 模型输入必须是 1×64
    model_input = processed.reshape(1, 64)

    # 13. 生成调试信息
    debug_info = {
        "原始图片尺寸": f"{w} × {h}",
        "裁剪后尺寸": f"{box_w} × {box_h}",
        "24×24 非零像素数": int(np.count_nonzero(medium)),
        "24×24 最大像素值": int(medium.max()),
        "8×8 非零像素数": int(np.count_nonzero(processed)),
        "8×8 最大像素值": float(processed.max())
    }

    crop_preview = Image.fromarray(crop)
    binary_preview = Image.fromarray(clean)
    resized_preview = Image.fromarray(medium)
    processed_preview = Image.fromarray(
        np.uint8(processed / 16.0 * 255.0)
    )

    return (
        model_input,
        crop_preview,
        binary_preview,
        resized_preview,
        processed_preview,
        processed,
        debug_info
    )


# ==========================================
# 预测函数
# ==========================================

def predict_digit(model, model_input):

    prediction = model.predict(model_input)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        confidence = float(np.max(probabilities)) * 100

    return prediction, confidence


# ==========================================
# 上传图片
# ==========================================

uploaded_file = st.file_uploader(
    "上传手写数字图片",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("原始图片")
        st.image(image, width=300)

        (
            model_input,
            crop_preview,
            binary_preview,
            resized_preview,
            processed_preview,
            processed,
            debug_info
        ) = preprocess_image(image)

        # ======================================
        # 预处理结果
        # ======================================

        st.divider()
        st.subheader("图片预处理结果")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Binary Image**")
            st.image(binary_preview, width=220)

        with col2:
            st.markdown("**Detected Handwriting**")
            st.image(crop_preview, width=220)

        with col3:
            st.markdown("**Resized Digit**")
            st.image(resized_preview, width=220)

        st.markdown("**Processed 8×8 Image**")
        st.image(processed_preview, width=300)

        # ======================================
        # 8×8 数字矩阵
        # ======================================

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        # ======================================
        # 调试信息
        # ======================================

        with st.expander("预处理调试信息"):
            for key, value in debug_info.items():
                st.write(f"**{key}：** {value}")

        # ======================================
        # 模型预测
        # ======================================

        st.divider()
        st.subheader("识别结果")

        col1, col2 = st.columns(2)

        if knn_model is not None:
            with col1:
                st.markdown("### KNN")

                try:
                    knn_prediction, knn_confidence = predict_digit(
                        knn_model,
                        model_input
                    )

                    st.metric(
                        "预测数字",
                        str(knn_prediction)
                    )

                    if knn_confidence is not None:
                        st.metric(
                            "模型置信度",
                            f"{knn_confidence:.2f}%"
                        )

                except Exception as e:
                    st.error(f"KNN 预测失败：{e}")

        if svm_model is not None:
            with col2:
                st.markdown("### SVM")

                try:
                    svm_prediction, svm_confidence = predict_digit(
                        svm_model,
                        model_input
                    )

                    st.metric(
                        "预测数字",
                        str(svm_prediction)
                    )

                    if svm_confidence is not None:
                        st.metric(
                            "模型置信度",
                            f"{svm_confidence:.2f}%"
                        )

                except Exception as e:
                    st.error(f"SVM 预测失败：{e}")

    except Exception as e:
        st.error(f"图片处理失败：{e}")

else:
    st.info("请先上传一张手写数字图片。")


# ==========================================
# 模型评估信息（如果存在）
# ==========================================

metrics_path = BASE_DIR / "metrics.json"

if metrics_path.exists():

    with st.expander("模型训练评估信息"):

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            st.json(metrics)

        except Exception as e:
            st.warning(f"无法读取 metrics.json：{e}")
