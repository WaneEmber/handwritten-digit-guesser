
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

    # 1. 转换成灰度图
    rgb = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    # 2. 检查图片边缘背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. 轻微降噪
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 4. 根据背景颜色选择正确的二值化方向
    if background_value >= 127:

        # 浅色背景，深色笔画
        threshold_type = cv2.THRESH_BINARY_INV
        threshold_c = 8

    else:

        # 深色背景，浅色笔画
        threshold_type = cv2.THRESH_BINARY
        threshold_c = -8

    # 5. 自适应阈值
    # 这一步不再使用之前容易误判背景的全局 Otsu 阈值
    min_dimension = min(height, width)

    block_size = min(51, min_dimension)

    if block_size % 2 == 0:
        block_size -= 1

    block_size = max(3, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        threshold_c
    )

    # 6. 统计前景像素比例
    foreground_ratio = (
        np.count_nonzero(binary) / binary.size
    )

    # 7. 连通区域检测
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError(
            "没有检测到手写数字，请检查图片。"
        )

    # 8. 排除图片边缘的噪声区域
    candidates = []

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

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
            "没有找到位于图片内部的有效笔画。"
        )

    # 9. 找最大主体区域
    largest = max(
        candidates,
        key=lambda item: item["area"]
    )

    largest_area = largest["area"]

    lx = largest["x"]
    ly = largest["y"]
    lw = largest["w"]
    lh = largest["h"]

    # 10. 只保留主体附近的有效笔画
    # 避免把远处的背景噪声也放进数字
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

    min_component_area = max(
        8,
        int(largest_area * 0.01)
    )

    for item in candidates:

        i = item["id"]

        x = item["x"]
        y = item["y"]

        w = item["w"]
        h = item["h"]

        area = item["area"]

        # 面积太小的组件不保留
        if area < min_component_area:
            continue

        # 判断是否靠近数字主体
        near_main = not (
            x + w < expanded_x1 or
            x > expanded_x2 or
            y + h < expanded_y1 or
            y > expanded_y2
        )

        if near_main:

            clean[labels == i] = 255

            kept_components += 1

    if np.count_nonzero(clean) == 0:
        raise ValueError(
            "没有找到有效的数字笔画。"
        )

    # 11. 获取真正的数字边界
    points = cv2.findNonZero(clean)

    x, y, box_width, box_height = cv2.boundingRect(points)

    # 如果边界仍然接近整张照片，不继续生成错误预测
    if (
        box_width > width * 0.85 or
        box_height > height * 0.85
    ):

        raise ValueError(
            "背景仍被识别为数字。请使用背景更均匀、数字更清晰的图片。"
        )

    # 12. 裁剪数字，添加少量黑色边缘
    padding = max(
        3,
        int(max(box_width, box_height) * 0.08)
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)

    crop = clean[y1:y2, x1:x2]

    crop_height, crop_width = crop.shape

    # 13. 保持比例，缩放到最大 20×20
    target_size = 20

    scale = target_size / max(
        crop_width,
        crop_height
    )

    new_width = max(
        1,
        min(
            target_size,
            int(round(crop_width * scale))
        )
    )

    new_height = max(
        1,
        min(
            target_size,
            int(round(crop_height * scale))
        )
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

    # 14. 放进 28×28 画布中央
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

    # 15. 缩小到 8×8
    small = cv2.resize(
        canvas28,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # 16. 转换成 0–16
    # 不再使用最大值归一化，保留真实的灰度比例
    processed = (
        small.astype(np.float32) / 255.0 * 16.0
    )

    processed = np.clip(
        processed,
        0,
        16
    ).astype(np.float32)

    # 17. 检查有效像素
    nonzero_pixels = int(
        np.count_nonzero(processed)
    )

    if nonzero_pixels < 5:

        raise ValueError(
            "转换后的 8×8 像素太少，数字细节已经丢失。"
        )

    # 18. 模型输入
    model_input = processed.reshape(1, 64)

    # 19. 预览图片
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

    # 20. 调试信息
    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "背景亮度": round(background_value, 2),
        "自适应阈值窗口": block_size,
        "前景像素比例": f"{foreground_ratio * 100:.2f}%",
        "原始连通区域数量": count - 1,
        "保留的连通区域数量": kept_components,
        "最大笔画区域面积": largest_area,
        "数字边界尺寸": f"{box_width} × {box_height}",
        "实际裁剪尺寸": f"{crop_width} × {crop_height}",
        "缩放后尺寸": f"{new_width} × {new_height}",
        "28×28 非零像素数": int(np.count_nonzero(canvas28)),
        "8×8 非零像素格数": nonzero_pixels,
        "8×8 最大像素值": round(float(processed.max()), 2)
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
# 预测函数
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
        # 预处理结果
        # ==================================

        st.divider()

        st.subheader("图片预处理结果")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Binary Image**")

            st.image(
                binary_preview,
                width=300
            )

        with col2:

            st.markdown("**Detected Handwriting（裁剪后）**")

            st.image(
                crop_preview,
                width=300
            )

        st.divider()

        st.markdown("### Resized Digit（28×28）")

        st.image(
            resized_preview,
            width=280
        )

        st.markdown("### Processed 8×8 Image")

        st.image(
            processed_preview,
            width=400
        )

        st.markdown("### 8×8 Pixel Grid")

        st.image(
            grid_preview,
            width=460
        )

        # ==================================
        # 8×8 矩阵
        # ==================================

        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            processed.astype(int),
            use_container_width=True
        )

        # ==================================
        # 调试信息
        # ==================================

        with st.expander(
            "预处理调试信息",
            expanded=True
        ):

            for key, value in debug.items():

                st.write(
                    f"**{key}：** {value}"
                )

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

                    st.error(
                        f"KNN 预测失败：{e}"
                    )

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

                    st.error(
                        f"SVM 预测失败：{e}"
                    )

    except Exception as e:

        st.error(
            f"图片处理失败：{e}"
        )

else:

    st.info("请先上传一张手写数字图片。")


# ======================================
# 模型评估信息
# ======================================

metrics_path = BASE_DIR / "metrics.json"

if metrics_path.exists():

    with st.expander("模型训练评估信息"):

        try:

            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as f:

                st.json(json.load(f))

        except Exception as e:

            st.warning(
                f"无法读取 metrics.json：{e}"
            )
