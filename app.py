
import streamlit as st
import numpy as np
import cv2
import joblib
import json

from pathlib import Path
from PIL import Image, ImageDraw


# ======================================
# 页面设置
# ======================================

st.set_page_config(
    page_title="Handwritten Digit Guesser",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Digit Guesser")
st.write("上传手写数字图片，使用 KNN 和 SVM 进行识别。")

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
# 8×8 像素网格预览
# ======================================

def make_pixel_grid(matrix):
    cell_size = 55
    margin = 4
    canvas_size = cell_size * 8 + margin * 2

    result = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        (25, 25, 25)
    )

    draw = ImageDraw.Draw(result)

    for row in range(8):
        for col in range(8):
            value = int(matrix[row, col])
            brightness = int(value / 16 * 255)

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
# 图片预处理
# ======================================

def preprocess_image(image):

    # 1. 转为灰度图
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # 2. 检查图片边缘，判断背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. 根据背景亮度统一笔画方向
    if background_value > 127:
        contrast = 255 - gray
    else:
        contrast = gray.copy()

    # 4. Otsu 自动二值化
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. 再次检查二值图边缘
    # 确保背景为黑色，笔画为白色
    binary_border = np.concatenate([
        binary[0, :],
        binary[-1, :],
        binary[:, 0],
        binary[:, -1]
    ])

    if np.median(binary_border) > 127:
        binary = 255 - binary

    # 6. 连接轻微断开的笔画
    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 7. 检测连通区域
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError("没有检测到有效的手写数字。")

    # 找到最大连通区域的面积
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(np.max(areas))

    # 根据主体大小过滤小噪点
    min_area = max(3, int(largest_area * 0.005))

    clean = np.zeros_like(binary)

    kept_components = 0

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255
            kept_components += 1

    if np.count_nonzero(clean) == 0:
        clean = binary.copy()

    # 8. 获取真正的数字边界
    points = cv2.findNonZero(clean)

    if points is None:
        raise ValueError("没有找到数字轮廓。")

    x, y, box_width, box_height = cv2.boundingRect(points)

    # 9. 仅保留很小的边缘空间
    padding = max(
        1,
        int(max(box_width, box_height) * 0.02)
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop = clean[y1:y2, x1:x2]

    # 10. 保持宽高比例，缩放至 7×7 范围
    crop_height, crop_width = crop.shape

    scale = 7.0 / max(crop_width, crop_height)

    new_width = max(
        1,
        min(7, int(round(crop_width * scale)))
    )

    new_height = max(
        1,
        min(7, int(round(crop_height * scale)))
    )

    if crop_width > 7 or crop_height > 7:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_NEAREST

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=interpolation
    )

    # 11. 放入 8×8 画布中央
    small = np.zeros((8, 8), dtype=np.uint8)

    start_x = (8 - new_width) // 2
    start_y = (8 - new_height) // 2

    small[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    # 12. 映射为 0–16
    max_value = float(small.max())

    if max_value <= 0:
        raise ValueError("8×8 图像没有有效笔画。")

    processed = (
        small.astype(np.float32) / max_value * 16.0
    )

    processed = np.clip(processed, 0, 16)

    model_input = processed.reshape(1, 64)

    # 13. 预览图
    binary_preview = Image.fromarray(clean)

    crop_preview = Image.fromarray(crop)

    # 最近邻放大，不模糊像素
    resized_preview = Image.fromarray(small).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    processed_display = np.uint8(
        processed / 16.0 * 255.0
    )

    grid_preview = make_pixel_grid(
        processed.astype(int)
    )

    processed_preview = Image.fromarray(
        processed_display
    ).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    # 14. 调试数据
    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "图片边缘背景亮度": round(background_value, 2),
        "原始检测到的区域": f"{box_width} × {box_height}",
        "实际裁剪尺寸": f"{crop_width} × {crop_height}",
        "缩放后尺寸": f"{new_width} × {new_height}",
        "保留的连通区域数量": kept_components,
        "裁剪图非零像素数": int(np.count_nonzero(crop)),
        "8×8 非零像素格数": int(np.count_nonzero(processed)),
        "8×8 最大像素值": float(processed.max())
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
# 模型预测
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
            binary_preview,
            crop_preview,
            resized_preview,
            processed_preview,
            grid_preview,
            processed,
            debug
        ) = preprocess_image(image)

        # ==================================
        # 预处理图片
        # ==================================

        st.divider()
        st.subheader("图片预处理结果")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Binary Image**")
            st.image(binary_preview, width=300)

        with col2:
            st.markdown("**Detected Handwriting（裁剪后）**")
            st.image(crop_preview, width=300)

        st.divider()

        st.markdown("### Resized Digit（8×8 放大预览）")
        st.image(resized_preview, width=400)

        st.markdown("### Processed 8×8 Image")
        st.image(processed_preview, width=400)

        st.markdown("### 8×8 Pixel Grid")
        st.image(grid_preview, width=460)

        # ==================================
        # 8×8 数字矩阵
        # ==================================

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        # ==================================
        # 调试信息
        # ==================================

        with st.expander("预处理调试信息", expanded=True):
            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

        # ==================================
        # 识别结果
        # ==================================

        st.divider()
        st.subheader("识别结果")

        col1, col2 = st.columns(2)

        if knn_model is not None:
            with col1:
                st.markdown("### KNN")

                try:
                    prediction, confidence = predict_digit(
                        knn_model,
                        model_input
                    )

                    st.metric("预测数字", str(prediction))

                    if confidence is not None:
                        st.metric(
                            "模型置信度",
                            f"{confidence:.2f}%"
                        )

                except Exception as e:
                    st.error(f"KNN 预测失败：{e}")

        if svm_model is not None:
            with col2:
                st.markdown("### SVM")

                try:
                    prediction, confidence = predict_digit(
                        svm_model,
                        model_input
                    )

                    st.metric("预测数字", str(prediction))

                    if confidence is not None:
                        st.metric(
                            "模型置信度",
                            f"{confidence:.2f}%"
                        )

                except Exception as e:
                    st.error(f"SVM 预测失败：{e}")

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
            st.warning(f"无法读取 metrics.json：{e}")
