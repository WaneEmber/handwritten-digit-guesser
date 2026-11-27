
def preprocess_image(image):

    # 1. Orientation and grayscale
    image = ImageOps.exif_transpose(image).convert("L")
    gray = np.asarray(image, dtype=np.uint8)

    h, w = gray.shape

    if min(h, w) < 3:
        raise ValueError("Image is too small.")

    # 2. Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Adaptive threshold
    min_side = min(h, w)

    block_size = min(
        51,
        min_side if min_side % 2 == 1 else min_side - 1
    )
    block_size = max(3, block_size)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7
    )

    # 4. Remove small noise, keep all meaningful components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    foreground = np.zeros_like(binary)

    min_area = max(8, int(h * w * 0.00003))

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area >= min_area:
            foreground[labels == i] = 255

    # 5. Find handwriting bounds
    ys, xs = np.where(foreground > 0)

    if len(xs) == 0:
        raise ValueError("No handwriting detected.")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cw = x_max - x_min + 1
    ch = y_max - y_min + 1

    crop = foreground[y_min:y_max + 1, x_min:x_max + 1]

    # 6. Add padding while preserving aspect ratio
    padding = max(2, int(max(cw, ch) * 0.15))
    side = max(cw, ch) + padding * 2

    square = np.zeros((side, side), dtype=np.uint8)

    sx = (side - cw) // 2
    sy = (side - ch) // 2

    square[sy:sy + ch, sx:sx + cw] = crop

    # 7. Resize to 24x24 first
    medium = cv2.resize(
        square,
        (24, 24),
        interpolation=cv2.INTER_NEAREST
    )

    # 8. Slightly strengthen strokes
    kernel = np.ones((2, 2), dtype=np.uint8)

    medium = cv2.dilate(
        medium,
        kernel,
        iterations=1
    )

    # 9. Reduce to 8x8
    small = cv2.resize(
        medium,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    # 10. Convert to 0-16
    small = small.astype(np.float32)

    max_value = float(small.max())

    if max_value <= 0:
        raise ValueError(
            "The digit disappeared during resizing. "
            "Please use a clearer image."
        )

    final_image = small / max_value * 16.0

    final_image = np.clip(final_image, 0, 16)

    # 11. Model input
    model_input = final_image.reshape(1, 64).astype(np.float32)

    # 12. Preview images
    preview_array = np.rint(
        final_image / 16.0 * 255
    ).astype(np.uint8)

    preview = Image.fromarray(preview_array)

    crop_preview = Image.fromarray(crop)

    binary_preview = Image.fromarray(binary)

    resized_preview = Image.fromarray(medium)

    bbox = (x_min, y_min, cw, ch)

    return (
        model_input,
        crop_preview,
        preview,
        binary_preview,
        resized_preview,
        bbox
    )
