
import streamlit as st
import numpy as np
import cv2
import joblib
import json

from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Digit Guesser")
st.write("上传一张手写数字图片进行识别。")

BASE_DIR = Path(__file__).resolve().parent


# ======================================
# 加载模型
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
    st.error(f"模型加载失败：{e}")
    st.stop()


# ======================================
# 图片预处理
# ======================================

def preprocess_image(image):

    # 1. 转为灰度图
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape

    # 2. 根据边缘估算背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background = float(np.median(border))

    # 3. 提取与背景不同的笔画
    if background > 127:
        contrast = 255 - gray
    else:
        contrast = gray.copy()

    # 4. Otsu 二值化
    _, binary = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. 连接相邻笔画
    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 6. 删除小噪点
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    clean = np.zeros_like(binary)

    min_area = max(3, int(h * w * 0.0002))

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    if np.count_nonzero(clean) == 0:
        clean = binary.copy()

    # 7. 获取数字边界
    points = cv2.findNonZero(clean)

    if points is None:
        raise ValueError("没有检测到数字，请更换图片。")

    x, y, box_w, box_h = cv2.boundingRect(points)

    crop = clean[y:y + box_h, x:x + box_w]

    # 8. 保持宽高比例，缩放到最大 7×7
    scale = 7.0 / max(box_w, box_h)

    new_w = max(1, min(7, int(round(box_w * scale))))
    new_h = max(1, min(7, int(round(box_h * scale))))

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 9. 放进 8×8 画布中央
    small = np.zeros((8, 8), dtype=np.uint8)

    start_x = (8 - new_w) // 2
    start_y = (8 - new_h) // 2

    small[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = resized

    # 10. 转换到 0-16
    processed = small.astype(np.float32) / 255.0 * 16.0
    processed = np.clip(processed, 0, 16)

    model_input = processed.reshape(1, 64)

    # 11. 预览图
    crop_preview = Image.fromarray(crop)
    binary_preview = Image.fromarray(clean)

    # 最近邻放大，清楚显示像素结构
    resized_preview = Image.fromarray(small).resize(
        (280, 280),
        Image.Resampling.NEAREST
    )

    processed_preview = Image.fromarray(
        np.uint8(processed / 16.0 * 255.0)
    ).resize(
        (320, 320),
        Image.Resampling.NEAREST
    )

    debug = {
        "原始图片尺寸": f"{w} × {h}",
        "数字裁剪尺寸": f"{box_w} × {box_h}",
        "缩放后尺寸": f"{new_w} × {new_h}",
        "8×8 非零像素数": int(np.count_nonzero(small)),
        "8×8 最大像素值": int(small.max())
    }

    return (
        model_input,
        crop_preview,
        binary_preview,
        resized_preview,
        processed_preview,
        processed,
        debug
    )


# ======================================
# 预测
# ======================================

def predict_digit(model, model_input):

    prediction = model.predict(model_input)[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        confidence = float(np.max(probabilities)) * 100

    return prediction, confidence


# ======================================
# 上传图片
# ======================================

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
            debug
        ) = preprocess_image(image)

        st.divider()
        st.subheader("图片预处理结果")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Binary Image**")
            st.image(binary_preview, width=250)

            st.markdown("**Detected Handwriting**")
            st.image(crop_preview, width=250)

        with col2:
            st.markdown("**Resized Digit (8×8 放大预览)**")
            st.image(resized_preview, width=280)

            st.markdown("**Processed 8×8 Image**")
            st.image(processed_preview, width=320)

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        with st.expander("预处理调试信息"):
            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

        st.divider()
        st.subheader("识别结果")

        col1, col2 = st.columns(2)

        if knn_model is not None:
            with col1:
                st.markdown("### KNN")

                prediction, confidence = predict_digit(
                    knn_model,
                    model_input
                )

                st.metric("预测数字", str(prediction))

                if confidence is not None:
                    st.metric("模型置信度", f"{confidence:.2f}%")

        if svm_model is not None:
            with col2:
                st.markdown("### SVM")

                prediction, confidence = predict_digit(
                    svm_model,
                    model_input
                )

                st.metric("预测数字", str(prediction))

                if confidence is not None:
                    st.metric("模型置信度", f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"图片处理失败：{e}")

else:
    st.info("请先上传一张手写数字图片。")


# ======================================
# 模型评估信息
# ======================================

metrics_path = BASE_DIR / "metrics.json"

if metrics_path.exists():
    with st.expander("模型训练评估信息"):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                st.json(json.load(f))
        except Exception as e:
            st.warning(f"无法读取评估信息：{e}")
