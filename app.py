
import streamlit as st
import numpy as np
import cv2
import joblib
import json

from pathlib import Path
from PIL import Image, ImageDraw

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
# 8×8 像素网格
# ======================================

def make_pixel_grid(matrix):
    cell_size = 55
    margin = 5
    size = cell_size * 8 + margin * 2

    result = Image.new("RGB", (size, size), (30, 30, 30))
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
# 检测数字内部的封闭区域
# ======================================

def find_enclosed_holes(binary_image):

    # 黑白反转：原来的黑色背景变成白色
    inverse = 255 - binary_image

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=8
    )

    height, width = binary_image.shape
    holes = []

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 接触裁剪边缘的区域属于外部背景，不是洞
        touches_edge = (
            x == 0 or
            y == 0 or
            x + w >= width or
            y + h >= height
        )

        if not touches_edge and area >= 3:
            cx, cy = centroids[i]
            holes.append((float(cx), float(cy), int(area)))

    return holes


# ======================================
# 图片预处理
# ======================================

def preprocess_image(image):

    # 1. 转换成灰度图
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # 2. 获取背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. 让数字笔画变亮、背景变暗
    if background_value >= 127:
        contrast = 255 - gray.astype(np.float32)
    else:
        contrast = gray.astype(np.float32)

    contrast = np.clip(contrast, 0, 255).astype(np.uint8)

    # 4. 轻微降噪
    blurred = cv2.GaussianBlur(
        contrast,
        (3, 3),
        0
    )

    # 5. 自动阈值
    otsu_value, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold_value = max(12, float(otsu_value))

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # 注意：这里不做 MORPH_CLOSE，避免封闭区域被填掉

    # 6. 删除小噪点
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        raise ValueError("没有检测到数字笔画。")

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    min_area = max(2, int(largest_area * 0.002))

    clean = np.zeros_like(binary)
    kept_components = 0

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255
            kept_components += 1

    if np.count_nonzero(clean) == 0:
        raise ValueError("去噪后没有剩余笔画。")

    # 7. 获取数字边界
    points = cv2.findNonZero(clean)

    x, y, box_width, box_height = cv2.boundingRect(points)

    # 8. 裁剪数字，加少量边缘
    padding = max(3, int(max(box_width, box_height) * 0.08))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop = clean[y1:y2, x1:x2]

    crop_height, crop_width = crop.shape

    # 9. 在原始裁剪图上检测封闭区域
    holes = find_enclosed_holes(crop)

    # 10. 保持比例，直接缩放到最大 7×7
    target_size = 7

    scale = target_size / max(crop_width, crop_height)

    new_width = max(
        1,
        min(7, int(round(crop_width * scale)))
    )

    new_height = max(
        1,
        min(7, int(round(crop_height * scale)))
    )

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # 11. 放入 8×8 中央
    small = np.zeros((8, 8), dtype=np.uint8)

    start_x = (8 - new_width) // 2
    start_y = (8 - new_height) // 2

    small[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    # 12. 保留原图中的封闭区域
    preserved_holes = []

    for cx, cy, area in holes:

        # 把原图洞的位置映射到 8×8
        mapped_x = (
            start_x +
            (cx + 0.5) * new_width / crop_width -
            0.5
        )

        mapped_y = (
            start_y +
            (cy + 0.5) * new_height / crop_height -
            0.5
        )

        col = int(round(mapped_x))
        row = int(round(mapped_y))

        col = max(0, min(7, col))
        row = max(0, min(7, row))

        # 将洞中心像素恢复为黑色
        small[row, col] = 0

        preserved_holes.append((row, col, area))

    # 13. 转换到 0–16
    processed = (
        small.astype(np.float32) / 255.0 * 16.0
    )

    processed = np.clip(processed, 0, 16)

    model_input = processed.reshape(1, 64)

    # 14. 生成预览
    binary_preview = Image.fromarray(clean)
    crop_preview = Image.fromarray(crop)

    resized_preview = Image.fromarray(small).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    processed_display = np.uint8(
        processed / 16 * 255
    )

    processed_preview = Image.fromarray(
        processed_display
    ).resize(
        (400, 400),
        Image.Resampling.NEAREST
    )

    grid_preview = make_pixel_grid(
        processed.astype(int)
    )

    # 15. 调试信息
    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "背景亮度": round(background_value, 2),
        "二值化阈值": round(threshold_value, 2),
        "数字边界尺寸": f"{box_width} × {box_height}",
        "实际裁剪尺寸": f"{crop_width} × {crop_height}",
        "8×8 缩放尺寸": f"{new_width} × {new_height}",
        "检测到的封闭区域数量": len(holes),
        "保留的封闭区域位置": str(preserved_holes),
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
# 页面主体
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

        st.divider()
        st.subheader("图片预处理结果")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Binary Image**")
            st.image(binary_preview, width=300)

        with col2:
            st.markdown("**Detected Handwriting**")
            st.image(crop_preview, width=300)

        st.divider()

        st.markdown("### Resized Digit（8×8 放大预览）")
        st.image(resized_preview, width=400)

        st.markdown("### Processed 8×8 Image")
        st.image(processed_preview, width=400)

        st.markdown("### 8×8 Pixel Grid")
        st.image(grid_preview, width=460)

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        with st.expander("预处理调试信息", expanded=True):
            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

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
