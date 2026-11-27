
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
# 生成清晰的 8×8 像素网格
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
            brightness = int(np.clip(value / 16, 0, 1) * 255)

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
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # 2. 估算图片背景亮度
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 3. 平滑图片，用于检测数字边界
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 4. 根据背景选择二值化方向
    if background_value >= 127:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    # 5. 自适应阈值只用于检测数字位置
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

    # 6. 连通区域检测
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        raise ValueError("没有检测到手写数字。")

    candidates = []

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 排除与图片边缘相连的区域
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
            "没有找到数字主体，请检查图片背景和边缘。"
        )

    # 7. 找到面积最大的主体区域
    largest = max(
        candidates,
        key=lambda item: item["area"]
    )

    largest_area = largest["area"]

    lx = largest["x"]
    ly = largest["y"]
    lw = largest["w"]
    lh = largest["h"]

    # 8. 保留主体附近的笔画
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
        raise ValueError("没有找到有效的数字笔画。")

    # 9. 获取数字边界
    points = cv2.findNonZero(clean)

    x, y, box_width, box_height = cv2.boundingRect(points)

    # 防止整张背景被误当成数字
    if (
        box_width > width * 0.85 or
        box_height > height * 0.85
    ):
        raise ValueError(
            "检测区域仍然包含大面积背景，请更换图片。"
        )

    # 10. 增加少量边缘
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
    # 关键修改：保留原始灰度笔画
    # ==================================

    # 11. 估算局部背景，减少照片背景不均匀的影响
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

    # 12. 只保留检测到的笔画区域，但保留灰度深浅
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

    # 13. 对笔画强度进行归一化
    positive_pixels = crop_gray[crop_gray > 0]

    if len(positive_pixels) == 0:
        raise ValueError("没有提取到有效的灰度笔画。")

    intensity_reference = float(
        np.percentile(positive_pixels, 95)
    )

    if intensity_reference < 1:
        raise ValueError("笔画亮度不足，无法生成模型输入。")

    crop_gray = (
        crop_gray / intensity_reference * 255.0
    )

    crop_gray = np.clip(
        crop_gray,
        0,
        255
    ).astype(np.uint8)

    # ==================================
    # 缩放到 8×8
    # ==================================

    # 14. 保持宽高比例，缩放到最大 7×7
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

    # 15. 放到 8×8 画布中央
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

    # 16. 转为 sklearn Digits 所使用的 0–16 范围
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
    # 预览图片
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
    # 调试信息
    # ==================================

    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "背景亮度": round(background_value, 2),
        "数字边界尺寸": f"{box_width} × {box_height}",
        "实际裁剪尺寸": f"{crop_width} × {crop_height}",
        "缩放后尺寸": f"{new_width} × {new_height}",
        "保留的连通区域数量": kept_components,
        "灰度笔画参考亮度": round(intensity_reference, 2),
        "8×8 非零像素格数": int(np.count_nonzero(processed)),
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
# 模型预测
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

        # 预处理结果
        st.divider()
        st.subheader("图片预处理结果")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Binary Image（边界检测）**")
            st.image(binary_preview, width=300)

        with col2:

            st.markdown("**Detected Handwriting（灰度笔画）**")
            st.image(crop_preview, width=300)

        st.divider()

        st.markdown("### Resized Digit（8×8 放大预览）")
        st.image(resized_preview, width=400)

        st.markdown("### Processed 8×8 Image")
        st.image(processed_preview, width=400)

        st.markdown("### 8×8 Pixel Grid")
        st.image(grid_preview, width=460)

        # 矩阵
        st.subheader("8×8 Pixel Matrix")

        st.dataframe(
            np.rint(processed).astype(int),
            use_container_width=True
        )

        # 调试信息
        with st.expander("预处理调试信息", expanded=True):

            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

        # 模型预测
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

                st.metric(
                    "预测数字",
                    str(prediction)
                )

                if confidence is not None:

                    st.metric(
                        "模型置信度",
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
                    "预测数字",
                    str(prediction)
                )

                if confidence is not None:

                    st.metric(
                        "模型置信度",
                        f"{confidence:.2f}%"
                    )

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

            with open(
                metrics_path,
                "r",
                encoding="utf-8"
            ) as f:

                st.json(json.load(f))

        except Exception as e:

            st.warning(f"无法读取 metrics.json：{e}")
