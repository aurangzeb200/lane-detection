import numpy as np

def compute_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')
    return (y2 - y1) / (x2 - x1)

def filter_and_group_segments(segments, image_width, slope_threshold=0.1):
    left_points = []
    right_points = []
    
    half_width = image_width // 2
    buffer = 50

    for x1, y1, x2, y2 in segments:
        slope = compute_slope(x1, y1, x2, y2)
        
        if abs(slope) < slope_threshold:
            continue
            
        if slope < 0:
            if x1 < (half_width + buffer) and x2 < (half_width + buffer):
                left_points.extend([[x1, y1], [x2, y2]])
            
        elif slope > 0:
            if x1 > (half_width - buffer) and x2 > (half_width - buffer):
                right_points.extend([[x1, y1], [x2, y2]])

    return np.array(left_points), np.array(right_points)

def linear_regression_fit(points):
    if len(points) == 0:
        return None, None
    
    x = points[:, 0]
    y = points[:, 1]
    n = len(points)
    
    sum_y = np.sum(y)
    sum_x = np.sum(x)
    sum_y2 = np.sum(y ** 2)
    sum_yx = np.sum(y * x)

    denominator = (n * sum_y2 - sum_y ** 2)
    
    if denominator == 0:
        return None, None

    m_inv = (n * sum_yx - sum_y * sum_x) / denominator
    b_inv = (sum_x - m_inv * sum_y) / n
    
    return m_inv, b_inv

def draw_line(image, x1, y1, x2, y2, color, thickness):
    h, w = image.shape[:2]
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    while True:
        for i in range(-thickness // 2, thickness // 2 + 1):
            for j in range(-thickness // 2, thickness // 2 + 1):
                draw_x = x1 + i
                draw_y = y1 + j
                
                if 0 <= draw_x < w and 0 <= draw_y < h:
                    image[draw_y, draw_x] = color

        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def draw_lane_line(image, m_inv, b_inv, color=(0,0,255), thickness=10):
    if m_inv is None or b_inv is None:
        return image
        
    h, w = image.shape[:2]
    y1 = h
    y2 = int(h * 0.6)
    
    x1 = int(m_inv * y1 + b_inv)
    x2 = int(m_inv * y2 + b_inv)
    
    draw_line(image, x1, y1, x2, y2, color, thickness)
    return image