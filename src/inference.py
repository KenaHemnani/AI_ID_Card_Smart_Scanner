import cv2
import numpy as np
import math
from ultralytics import YOLO
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import logging
import sys

def get_4_corners(model, image_path, epsilon_factor=0.02):
    # model = YOLO(model_path)
    results = model(image_path)

    image_bgr = cv2.imread(image_path)  # Original image in BGR
    H, W, _ = image_bgr.shape
    contour_vis = image_bgr.copy()  # For contour visualization

    # Define colors for each arrow (BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0)   # Yellow
    ]

    corners = None  # To store corners from the best mask

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()
        
        for mask in masks:
            mask_resized = cv2.resize(mask, (W, H))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw all contours on contour_vis image
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 255), 2)  # Yellow contours
            # print(contours)
            for contour in contours:
                if len(contour) < 4:
                    continue

                peri = cv2.arcLength(contour, True)
                epsilon = epsilon_factor * peri
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    corners = approx.reshape(4, 2)

                    for i in range(4):
                        pt1 = tuple(corners[i])
                        pt2 = tuple(corners[(i + 1) % 4])
                        color = colors[i % len(colors)]

                        # Draw arrowed line
                        cv2.arrowedLine(image_bgr, pt1, pt2, color, 8, tipLength=0.05)

                        # Draw corner index
                        cv2.putText(
                            image_bgr,
                            str(i),
                            pt1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA
                        )

    if corners is None:
        logging.error("No 4-corner mask found.")
        # return None

    # Save image with labeled corners
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Remove axes
    # No title
    plt.savefig('output/output.jpg', bbox_inches='tight', pad_inches=0)
    logging.info('Prediction output image saved at output/output.jpg')
    plt.close()  # Close the figure to free memory

    return corners


# Calculate angle of the line formed by two points
def calculate_angle(pt1, pt2):
    # if pt1[0] > pt2[0]:
    #     pt1, pt2 = pt2, pt1
    delta_y = pt2[1] - pt1[1]
    delta_x = pt2[0] - pt1[0]
    angle_rad = (-1) * math.atan2(delta_y, delta_x)  # returns angle in radians
    angle_deg = math.degrees(angle_rad)  # Convert to degrees
    return angle_deg

def reorder_points_based_on_angle(points, angle):

    # Normalize angle to range [0, 180)
    normalized_angle = abs(angle) % 180

    # if abs(normalized_angle) < 20:  # Near 0°
    #     return points[[1, 0, 3, 2]]  # top-left, top-right, bottom-right, bottom-left
    if abs(normalized_angle - 90) < 20:  # Near 90°
        rotation_angle = -(normalized_angle - 90)
        return points[[0, 3, 2, 1]], rotation_angle  # rotated order
    else:
        rotation_angle = -(normalized_angle)
        return points[[1, 0, 3, 2]], rotation_angle 


def visualize_angle(image, pt1, pt2, angle, axis_length=500):
    # Copy the image to avoid modifying original
    image_viz = image.copy()

    # Draw the angle line (Red)
    cv2.arrowedLine(image_viz, tuple(pt1), tuple(pt2), (0, 0, 255), 8, tipLength=0.1)  # Red

    # Add angle text
    mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.putText(
        image_viz,
        f'{angle:.2f} deg',
        mid_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Draw positive X-axis (Purple)
    pt_x = (pt1[0] + axis_length, pt1[1])
    cv2.arrowedLine(image_viz, tuple(pt1), pt_x, (128, 0, 128), 5, tipLength=0.1)

    pt_y = (pt1[0], pt1[1] - axis_length)
    cv2.arrowedLine(image_viz, tuple(pt1), pt_y, (0, 165, 255), 5, tipLength=0.1)

    return image_viz



def unskew(image, pts_src):
    pts_src = np.array(pts_src, dtype="float32")

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    widthA = distance(pts_src[2], pts_src[3])
    widthB = distance(pts_src[1], pts_src[0])
    maxWidth = int(max(widthA, widthB))

    heightA = distance(pts_src[1], pts_src[2])
    heightB = distance(pts_src[0], pts_src[3])
    maxHeight = int(max(heightA, heightB))

    pts_dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def run_inference(model, img_path):
    # Get the 4 corners of the detected mask
    points = get_4_corners(model, img_path)

    # Calculate the angle of the red axis w.r.t positive X-axis
    angle = calculate_angle(points[2], points[3])
    # print(f"Angle of red axis with horizontal axis: {angle:.2f} degrees")
    logging.info(f"Angle of red axis with horizontal axis: {angle:.2f} degrees")

    reordered_points, rotation_angle = reorder_points_based_on_angle(points, angle)
    if rotation_angle <= 0:
        # print(f"Image i.e., Red axis rotated by {abs(rotation_angle)} clock wise")
        logging.info(f"Image i.e., Red axis rotated by {abs(rotation_angle)} clock wise")
    else:
        logging.info(f"Image i.e., Red axis rotated by {rotation_angle} anti-clock wise")
        # print(f"Image i.e., Red axis rotated by {rotation_angle} anti-clock wise")

    # Visualize the angle on the image
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for better visualization
    image_with_angle = visualize_angle(image.copy(), points[2], points[3], angle)
    # Save the final image with the angle visualization
    angle_visualized_filename = 'output/angle_visualized_output.jpg'
    cv2.imwrite(angle_visualized_filename, image_with_angle)
    logging.info(f'Angle visualization Image saved at {angle_visualized_filename}')

    # Unskew the image based on the ordered points
    cropped_unskewed_img = unskew(image, reordered_points)

    # Save the final unskewed image
    filename = 'output/corrected_img.jpg'
    cv2.imwrite(filename, cropped_unskewed_img)
    logging.info(f'Cropped and Unskewed Image saved at {filename}')

    return cropped_unskewed_img, rotation_angle

if __name__ == "__main__":

    # Set up basic logging config
    logging.basicConfig(
        level=logging.INFO,  # Set log level
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    model_path = 'best_doc_seg_yolo.pt'

    if len(sys.argv) < 2:
        logging.error("Usage: python your_script.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    model = YOLO(model_path)

    cropped_unskewed_img, rotation_angle = run_inference(model, img_path)

    logging.info(f"Rotation angle: {rotation_angle}")

