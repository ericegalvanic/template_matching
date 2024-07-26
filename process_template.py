import os
import numpy as np
import cv2 as cv
import logging
import boto3
import tempfile
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Predefined bright colors
BRIGHT_COLORS = [
    [0, 255, 0],      # Bright Green
    [255, 0, 0],      # Bright Red
    [128, 0, 128],    # Bright Purple
    [255, 165, 0],    # Bright Orange
    [255, 255, 0],    # Bright Yellow
    [0, 0, 255]       # Bright Blue
]

s3_client = boto3.client('s3')

def download_image(bucket, key):
    # Download an image from S3 to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3_client.download_file(Bucket=bucket, Key=key, Filename=tmp_file.name)
        return tmp_file.name

def upload_image(bucket, file_path, upload_key):
    # Upload an image to S3 and return the key
    s3_client.upload_file(Filename=file_path, Bucket=bucket, Key=upload_key)
    return upload_key

def apply_threshold(image, threshold_value=127):
    _, binary_image = cv.threshold(image, threshold_value, 255, cv.THRESH_BINARY)
    return binary_image

def rotate_image_90(image):
    return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

def rotate_image_270(image):
    return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

def rotate_image_180(image):
    return cv.rotate(image, cv.ROTATE_180)

def flip_image(image):
    return cv.flip(image, 1)

def create_mask(template_shape, ux, uy, lx, ly):
    mask = np.ones(template_shape, dtype=np.uint8) * 255
    if lx < 0: lx = template_shape[1] + lx
    if ly < 0: ly = template_shape[0] + ly
    cv.rectangle(mask, (ux, uy), (lx, ly), 0, thickness=-1)
    return mask

def process_template_matching(img_gray_thresh, template_thresh, matching_threshold, template_name, mask):
    boxes = []
    scores = []  # Store the scores for NMS
    transformations = [lambda x: x, flip_image, rotate_image_90, rotate_image_180, rotate_image_270]

    for transform in transformations:
        transformed_template = transform(template_thresh)
        transformed_mask = transform(mask)
        res = cv.matchTemplate(img_gray_thresh, transformed_template, cv.TM_CCORR_NORMED, mask=transformed_mask)

        loc = np.where(res >= matching_threshold)
        w, h = transformed_template.shape[::-1]
        for pt in zip(*loc[::-1]):
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
            scores.append(res[pt[::-1]])

    return boxes, scores, template_name

def non_max_suppression(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick

def process_template(s3_bucket, base_image_key, template_image_key, threshold, timeout_value, iteration, mask_key=None):

    # Download base and template images from S3
    base_image_path = download_image(s3_bucket, base_image_key)
    template_image_path = download_image(s3_bucket, template_image_key)

    # Load images using OpenCV
    img_rgb = cv.imread(base_image_path)
    assert img_rgb is not None, "Base image could not be loaded."
    logging.info("Base image loaded successfully.")

    template = cv.imread(template_image_path, cv.IMREAD_GRAYSCALE)
    assert template is not None, "Template image could not be loaded."
    logging.info("Template image loaded successfully.")

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    img_gray_thresh = apply_threshold(img_gray)
    template_thresh = apply_threshold(template)

    # Read mask if provided
    if mask_key:
        mask_path = download_image(s3_bucket, mask_key)
        with open(mask_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            row = next(reader)  # Assuming single row for the mask
            ux, uy, lx, ly = map(int, row[1:])
            mask = create_mask(template.shape, ux, uy, lx, ly)
        os.remove(mask_path)
    else:
        logging.info(f"No mask provided. Using default all-white mask.")
        mask = np.ones(template.shape, dtype=np.uint8) * 255  # Default mask (all white)

    # Perform template matching
    boxes, scores, template_name = process_template_matching(img_gray_thresh, template_thresh, threshold, template_image_key, mask)
    all_boxes_np = np.array(boxes)
    all_scores_np = np.array(scores)
    selected_indices = non_max_suppression(all_boxes_np, all_scores_np, 0.5)

    selected_boxes = all_boxes_np[selected_indices]
    selected_names = [template_name] * len(selected_boxes)

    annotation_counts = {}
    for box, name in zip(selected_boxes, selected_names):
        x1, y1, x2, y2 = box
        color = BRIGHT_COLORS[5]  # Use the last bright color for simplicity
        cv.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)  # Thickness set to 3

        if name not in annotation_counts:
            annotation_counts[name] = 0
        annotation_counts[name] += 1

    # Save and upload annotated image
    annotated_image_path = os.path.join(tempfile.gettempdir(), 'results.png')
    cv.imwrite(annotated_image_path, img_rgb)
    upload_key = f"1-{os.path.basename(base_image_key)}"
    upload_image(s3_bucket, annotated_image_path, upload_key)
    logging.info(f"Annotated image uploaded to S3 with key: {upload_key}")

    # Print the annotation counts
    for template_name, count in annotation_counts.items():
        logging.info(f"{template_name}: {count} instances detected")

if __name__ == "__main__":
    process_template(s3_bucket="eg-ai-dev", base_image_key="projects/d7ee8406-9399-4659-bb0a-be5040629010/base/base.jpg", template_image_key="projects/d7ee8406-9399-4659-bb0a-be5040629010/templates/meter.png", threshold=0.95, timeout_value=10, iteration=0, mask_key=None)