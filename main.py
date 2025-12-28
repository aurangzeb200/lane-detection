# RollNumber: BSAI23021
# Name: Aurangzeb
# Assignment: 04

import cv2
import os
import numpy as np
import argparse

from color_filter import apply_gaussian_blur, filter_lane_colors, convert_to_hsv
from edge_detection import canny_edge_detector
from region_of_interest import region_of_interest
from line_detection import linear_regression_fit, draw_lane_line, filter_and_group_segments,draw_line
from utils import combine_images
from hough_utils import hough_transform, hough_to_segments


def save(output_folder, image_path, file):    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename_only = os.path.basename(image_path)
    save_path = os.path.join(output_folder, filename_only)
    cv2.imwrite(save_path, file)


def process_image(image, image_path):
    hsv_image, binary_image = convert_to_hsv(image)
    save("HSV_files", image_path, hsv_image)

    blurred = apply_gaussian_blur(hsv_image)
    save("gaussian_files", image_path, blurred)

    filtered, rgb_filtered = filter_lane_colors(blurred, binary_image, image)
    save("binary_files", image_path, filtered)
    save("binary_rgb_files", image_path, rgb_filtered)

    edges = canny_edge_detector(image)
    canny_edge_detector_bitwise = cv2.bitwise_and(filtered, edges)
    save("canny_edge_detector_files", image_path, edges)
    save("canny_edge_detector_bitwise_files", image_path, canny_edge_detector_bitwise)

    cropped_edges, mask = region_of_interest(canny_edge_detector_bitwise)
    save("region_of_interest_files", image_path, cropped_edges)
    save("mask_files", image_path, mask)

    # Hough transform
    hough_lines = hough_transform(cropped_edges, rho_res=1, theta_res=1, threshold=15)

    # Make a blank image to visualize Hough lines
    hough_img = np.zeros_like(image)
    segments = hough_to_segments(hough_lines, cropped_edges)
    for x1, y1, x2, y2 in segments:
        draw_line(hough_img, x1, y1, x2, y2, color=(0, 255, 255), thickness=2)
    save("hough_lines_visual", image_path, hough_img)

    h, w = cropped_edges.shape[:2]
    left_points, right_points = filter_and_group_segments(segments, image_width=w)

    left_m, left_b = linear_regression_fit(left_points)
    right_m, right_b = linear_regression_fit(right_points)

    # Draw lanes on a blank image
    lanes_img = np.zeros((h, w, 3), dtype=np.uint8)
    draw_lane_line(lanes_img, left_m, left_b, color=(0, 0, 255))
    draw_lane_line(lanes_img, right_m, right_b, color=(0, 255, 0))
    save("lanes_only", image_path, lanes_img)

    # Final overlay
    final_image = combine_images(image, lanes_img)
    save("final_output", image_path, final_image)

    return final_image

def main():
    parser = argparse.ArgumentParser(description="Lane Departure Detection System")
    parser.add_argument('--input_folder', required=True, help="Path to input images folder")
    parser.add_argument('--output_folder', required=True, help="Path to save output results")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error loading {filename}")
                continue

            result = process_image(image, img_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            print(f"Processed and saved: {output_path}")

    print("All images processed successfully!")


if __name__ == "__main__":
    main()
