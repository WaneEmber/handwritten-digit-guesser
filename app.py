
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
# 生成清晰的 8×8 像素预览图
# ======================================

def make_pixel_grid(matrix):
    cell_size = 55
    margin = 4

    canvas_size = cell_size * 8 + margin * 2

    image = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        (30, 30, 30)
    )

    draw = ImageDraw.Draw(image)

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
                outline=(90, 90, 90),
                width=1
            )

    return image


# ======================================
# 图片预处理
# ======================================

def preprocess_image(image):

    # 1. 转换为灰度图
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    # 2. 判断背景颜色
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background_value = float(np.median(border))

    # 统一为：黑色背景、白色笔画
    if background_value > 127:
        ink = 255 - gray
    else:
        ink = gray.copy()

    # 3. 降低噪声
    blurred = cv2.GaussianBlur(
        ink,
        (3, 3),
        0
    )

    # 4. 自动二值化
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. 去除极小噪点，尽可能保留笔画
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    clean = np.zeros_like(binary)

    min_area = max(2, int(height * width * 0.00001))

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean[labels == i] = 255

    if np.count_nonzero(clean) == 0:
        clean = binary.copy()

    # 6. 获取数字边界
    points = cv2.findNonZero(clean)

    if points is None:
        raise ValueError("没有检测到数字，请检查上传的图片。")

    x, y, box_width, box_height = cv2.boundingRect(points)

    crop = clean[
        y:y + box_height,
        x:x + box_width
    ]

    # 7. 保持比例，放到 8×8 画布内
    target_size = 7

    scale = target_size / max(box_width, box_height)

    new_width = max(
        1,
        min(target_size, int(round(box_width * scale)))
    )

    new_height = max(
        1,
        min(target_size, int(round(box_height * scale)))
    )

    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_NEAREST

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=interpolation
    )

    # 8. 放到 8×8 画布中央
    small = np.zeros((8, 8), dtype=np.uint8)

    start_x = (8 - new_width) // 2
    start_y = (8 - new_height) // 2

    small[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    # 9. 关键修改：归一化到 0–16
    # 避免缩小后的灰度值太低，导致图像几乎全黑
    max_value = float(small.max())

    if max_value <= 0:
        raise ValueError("缩放后的 8×8 图像没有有效像素。")

    processed = np.rint(
        small.astype(np.float32) / max_value * 16
    )

    processed = np.clip(
        processed,
        0,
        16
    ).astype(np.float32)

    # 10. 模型输入
    model_input = processed.reshape(1, 64)

    # 11. 准备显示图片
    binary_preview = Image.fromarray(clean)
    crop_preview = Image.fromarray(crop)

    # 将 8×8 图像放大，使用最近邻插值
    resized_preview = Image.fromarray(
        (processed * 255 / 16).astype(np.uint8)
    ).resize(
        (440, 440),
        Image.Resampling.NEAREST
    )

    # 使用带网格的方式显示每个像素
    grid_preview = make_pixel_grid(processed.astype(int))

    # 12. 调试数据
    debug = {
        "原始图片尺寸": f"{width} × {height}",
        "数字裁剪尺寸": f"{box_width} × {box_height}",
        "缩放后尺寸": f"{new_width} × {new_height}",
        "裁剪区域非零像素数": int(np.count_nonzero(crop)),
        "8×8 非零像素格数": int(np.count_nonzero(processed)),
        "8×8 最大像素值": float(processed.max()),
        "8×8 最小非零像素值": float(processed[processed > 0].min())
    }

    return (
        model_input,
        binary_preview,
        crop_preview,
        resized_preview,
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
            st.image(binary_preview, width=300)

        with col2:
            st.markdown("**Detected Handwriting**")
            st.image(crop_preview, width=300)

        st.divider()

        st.markdown("### Resized Digit (8×8 放大预览)")
        st.image(resized_preview, width=440)

        st.markdown("### Processed 8×8 Image（像素网格）")
        st.image(grid_preview, width=460)

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

        with st.expander("预处理调试信息"):
            for key, value in debug.items():
                st.write(f"**{key}：** {value}")

        # ==================================
        # 预测结果
        # ==================================

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

                st.metric("预测数字", str(prediction))

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
            with open(metrics_path, "r", encoding="utf-8") as f:
                st.json(json.load(f))

        except Exception as e:
            st.warning(f"无法读取 metrics.json：{e}")
