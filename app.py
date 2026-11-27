
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
# 8×8 像素网格预览
# ======================================

def make_pixel_grid(matrix):

    cell_size = 55
    margin = 5
    canvas_size = cell_size * 8 + margin * 2

    result = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        (30, 30, 30)
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

    # 1. 转成灰度图
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # 2. 获取图片四周的背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. 计算像素与背景之间的亮度差
    # 无论白底黑字还是黑底白字，都让笔画变亮
    if background_value >= 127:
        ink_strength = background_value - gray.astype(np.float32)
    else:
        ink_strength = gray.astype(np.float32) - background_value

    ink_strength = np.clip(
        ink_strength,
        0,
        255
    ).astype(np.uint8)

    # 4. 平滑少量噪声
    blurred = cv2.GaussianBlur(
        ink_strength,
        (3, 3),
        0
    )

    # 5. Otsu 自动阈值，并设置最低阈值
    otsu_threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold_value = max(
        12,
        float(otsu_threshold)
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # 6. 检查是否把大片背景误当成笔画
    foreground_ratio = np.count_nonzero(binary) / binary.size

    if foreground_ratio > 0.55:

        # 提高阈值，只提取明显区别于背景的部分
        stronger_threshold = max(
            threshold_value,
            float(np.percentile(blurred, 85))
        )

        _, binary = cv2.threshold(
            blurred,
            stronger_threshold,
            255,
            cv2.THRESH_BINARY
        )

        threshold_value = stronger_threshold

    # 7. 确保图片边缘是背景
    binary_border = np.concatenate([
        binary[0, :],
        binary[-1, :],
        binary[:, 0],
        binary[:, -1]
    ])

    if np.median(binary_border) > 127:
        binary = 255 - binary

    foreground_ratio = np.count_nonzero(binary) / binary.size

    # 8. 检测连通区域
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        raise ValueError(
            "没有检测到数字笔画，请检查图片是否清晰。"
        )

    # 9. 删除很小的噪点
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(np.max(areas))

    min_area = max(
        2,
        int(largest_area * 0.005)
    )

    clean = np.zeros_like(binary)
    kept_components = 0

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255
            kept_components += 1

    if np.count_nonzero(clean) == 0:
        raise ValueError("去除噪点后没有剩余笔画。")

    # 10. 获取数字的真实边界
    points = cv2.findNonZero(clean)

    if points is None:
        raise ValueError("无法获取数字轮廓。")

    x, y, box_width, box_height = cv2.boundingRect(points)

    # 11. 裁剪数字，并只保留少量边缘
    padding = max(
        2,
        int(max(box_width, box_height) * 0.08)
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop = clean[y1:y2, x1:x2]

    crop_height, crop_width = crop.shape

    # 12. 保持比例，缩放到最大 20×20
    target_size = 20

    scale = target_size / max(
        crop_width,
        crop_height
    )

    new_width = max(
        1,
        min(target_size, int(round(crop_width * scale)))
    )

    new_height = max(
        1,
        min(target_size, int(round(crop_height * scale)))
    )

    if crop_width > target_size or crop_height > target_size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_NEAREST

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=interpolation
    )

    # 13. 放在 28×28 画布中央
    canvas28 = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    start_x = (28 - new_width) // 2
    start_y = (28 - new_height) // 2

    canvas28[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    # 14. 缩小到 8×8
    small = cv2.resize(
        canvas28,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # 15. 映射到 0–16
    max_value = float(small.max())

    if max_value <= 0:
        raise ValueError("8×8 图像没有有效笔画。")

    processed = (
        small.astype(np.float32) / max_value * 16.0
    )

    processed = np.clip(
        processed,
        0,
        16
    ).astype(np.float32)

    model_input = processed.reshape(1, 64)

    # 16. 生成预览图
    binary_preview = Image.fromarray(binary)
    crop_preview = Image.fromarray(crop)

    resized_preview = Image.fromarray(canvas28).resize(
        (280, 280),
        Image.Resampling.NEAREST
    )

    processed_display = np.uint8(
        processed / 16.0 * 255.0
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

    # 17. 调试数据
    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "背景亮度": round(background_value, 2),
        "使用的二值化阈值": round(threshold_value, 2),
        "前景像素比例": f"{foreground_ratio * 100:.2f}%",
        "数字边界尺寸": f"{box_width} × {box_height}",
        "实际裁剪尺寸": f"{crop_width} × {crop_height}",
        "缩放后尺寸": f"{new_width} × {new_height}",
        "保留的连通区域数量": kept_components,
        "28×28 非零像素数": int(np.count_nonzero(canvas28)),
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

        # 预处理图片
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

        st.markdown("### Resized Digit（28×28 放大预览）")
        st.image(resized_preview, width=280)

        st.markdown("### Processed 8×8 Image")
        st.image(processed_preview, width=400)

        st.markdown("### 8×8 Pixel Grid")
        st.image(grid_preview, width=460)

        # 8×8 数字矩阵
        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        # 调试数据
        with st.expander("预处理调试信息", expanded=True):
            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

        # 识别结果
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

                    st.metric(
                        "预测数字",
                        str(prediction)
                    )

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

                    st.metric(
                        "预测数字",
                        str(prediction)
                    )

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
