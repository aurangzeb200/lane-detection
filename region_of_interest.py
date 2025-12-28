import numpy as np

def region_of_interest(image):
    rows, cols = image.shape
    top_width = int(cols * 0.45)
    bottom_width = int(cols * 0.95)
    height = int(rows * 0.6)

    x_bottom_left  = (cols - bottom_width) // 2
    x_bottom_right = (cols + bottom_width) // 2
    x_top_left     = (cols - top_width) // 2
    x_top_right    = (cols + top_width) // 2
    y_bottom = rows
    y_top = rows - height

    mask = np.zeros_like(image, dtype=np.uint8)

    for y in range(rows):
        if y >= y_top:
            left_x  = int(x_top_left  + (x_bottom_left  - x_top_left)  * (y - y_top) / (y_bottom - y_top))
            right_x = int(x_top_right + (x_bottom_right - x_top_right) * (y - y_top) / (y_bottom - y_top))
            left_x = max(0, left_x)
            right_x = min(cols - 1, right_x)
            mask[y, left_x:right_x+1] = 1

    masked_image = image * mask
    return masked_image, mask
