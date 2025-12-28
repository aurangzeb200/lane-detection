import cv2
import numpy as np

def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_image = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    return hsv_image, binary_image

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gaussian_kernel = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()

    h, w, c = image.shape
    blurred = np.zeros_like(image, dtype=np.float64)

    padded = np.pad(image, ((k, k), (k, k), (0, 0)), mode='reflect')

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kernel_size, j:j+kernel_size, ch]
                blurred[i, j, ch] = np.sum(region * gaussian_kernel)

    return np.clip(blurred, 0, 255).astype(np.uint8)


def filter_lane_colors(hsv,binary_image,image):
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 70, 255])

    lower_yellow = np.array([15, 80, 90])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

    result = cv2.bitwise_and(binary_image, lane_mask)

    lane_mask_color = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    rgb_result = cv2.bitwise_and(image, lane_mask_color)

    return result , rgb_result
