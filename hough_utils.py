import numpy as np

def hough_transform(edge_image, rho_res=1, theta_res=1, threshold=10):
    rows, cols = edge_image.shape
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    diag_len = int(np.sqrt(rows**2 + cols**2))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    ys, xs = np.nonzero(edge_image)
    
    for x, y in zip(xs, ys):
        for t_idx in range(len(thetas)):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            r_idx = rho + diag_len
            accumulator[r_idx, t_idx] += 1

    lines = []
    r_idxs, t_idxs = np.where(accumulator >= threshold)
    for r, t in zip(r_idxs, t_idxs):
        lines.append((rhos[r], thetas[t]))
            
    return lines

def hough_to_segments(lines, image):
    segments = []
    h, w = image.shape[:2]
    
    y_bottom = h
    y_top = int(h * 0.6) 

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        
        if abs(a) < 0.001:
            continue

        x1 = int((rho - y_bottom * b) / a)
        y1 = y_bottom
        
        x2 = int((rho - y_top * b) / a)
        y2 = y_top

        if -200 <= x1 <= w + 200 and -200 <= x2 <= w + 200:
            segments.append([x1, y1, x2, y2])
        
    return segments
