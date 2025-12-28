# RollNumber: BSAI23021
# Name: Aurangzeb
# Assignment: 04

import numpy as np

def combine_images(base_image, overlay, alpha=0.8, beta=1.0, gamma=1.0):
    h, w, c = base_image.shape
    result = np.zeros_like(base_image, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                val = alpha * base_image[y, x, ch] + beta * overlay[y, x, ch] + gamma
                result[y, x, ch] = np.clip(int(val), 0, 255)
    
    return result
