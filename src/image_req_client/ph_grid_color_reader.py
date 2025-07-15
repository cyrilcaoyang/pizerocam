import os
import cv2
import numpy as np
from pathlib import Path
from google.cloud import vision
import itertools
import argparse

# --- CONFIG ---
# Path to the image to analyze
IMAGE_PATH = "photos-2025-03-26-pH/capture_20250714-182920_100100100.jpg"  # Change to your image path
OUTPUT_DIR = "photos-2025-03-26-pH"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: OCR Detection and Text Box Visualization ---

def detect_text_boxes(image_path):
    """Detect text and bounding boxes using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    image = cv2.imread(image_path)
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise RuntimeError("Failed to encode image for Vision API")
    vision_image = vision.Image(content=buffer.tobytes())
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations
    detections = []
    for text in texts[1:]:  # Skip the first, which is all text
        vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
        detections.append({
            'text': text.description,
            'bbox': vertices
        })
    return image, detections

def draw_text_boxes(image, detections, out_path):
    """Draw bounding boxes and text labels on the image and save it."""
    img = image.copy()
    for det in detections:
        bbox = np.array(det['bbox'], dtype=np.int32)
        cv2.polylines(img, [bbox], isClosed=True, color=(0,255,0), thickness=10)
        x, y = bbox[0]
        cv2.putText(img, det['text'], (x+150, y+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 10)
    cv2.imwrite(out_path, img)
    print(f"Saved labeled text box image to {out_path}")

def cluster_rows(detections, row_gap_thresh=40):
    """Cluster text boxes into rows based on y-coordinate proximity."""
    # Use the vertical center of each box
    for det in detections:
        bbox = det['bbox']
        det['y_center'] = int(np.mean([p[1] for p in bbox]))
        det['x_center'] = int(np.mean([p[0] for p in bbox]))
    # Sort by y_center
    detections_sorted = sorted(detections, key=lambda d: d['y_center'])
    rows = []
    for k, group in itertools.groupby(detections_sorted, key=lambda d: d['y_center']//row_gap_thresh):
        rows.append(list(group))
    return rows

def define_color_boxes(rows, delta_y_frac=0.4, width_frac=0.5):
    """For each number, define a color box below it."""
    if len(rows) < 2:
        print("Warning: Less than 2 rows detected. Color box height may be inaccurate.")
        row_delta = 40  # fallback
    else:
        # Use mean y distance between row centers
        row_delta = abs(np.mean([det['y_center'] for det in rows[1]]) - np.mean([det['y_center'] for det in rows[0]]))
    color_boxes = []
    for row in rows:
        # Sort by x
        row_sorted = sorted(row, key=lambda d: d['x_center'])
        # Compute average x distance between numbers
        if len(row_sorted) > 1:
            x_dists = [row_sorted[i+1]['x_center'] - row_sorted[i]['x_center'] for i in range(len(row_sorted)-1)]
            avg_x_dist = np.mean(x_dists)
        else:
            avg_x_dist = 40  # fallback
        for det in row_sorted:
            bbox = det['bbox']
            x_center = det['x_center']
            y_max = max([p[1] for p in bbox])
            width = int(avg_x_dist * width_frac)
            height = int(row_delta * delta_y_frac)
            x1 = int(x_center - width//2)
            x2 = int(x_center + width//2)
            y1 = int(y_max + row_delta * 0.4 - height//2)
            y2 = int(y1 + height)
            color_boxes.append({
                'ph_text': det['text'],
                'rect': (x1, y1, x2, y2),
                'x_center': x_center,
                'y_max': y_max
            })
    return color_boxes

def draw_color_boxes(image, detections, color_boxes, out_path=None):
    img = image.copy()
    for det in detections:
        bbox = np.array(det['bbox'], dtype=np.int32)
        cv2.polylines(img, [bbox], isClosed=True, color=(0,255,0), thickness=10)
        x, y = bbox[0]
        cv2.putText(img, det['text'], (x+150, y+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 10)
    for box in color_boxes:
        x1, y1, x2, y2 = box['rect']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 10)  # White outline
        cv2.putText(img, f"pH {box['ph_text']}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 10)  # White label, size 3
    if out_path:
        cv2.imwrite(out_path, img)
        print(f"Saved labeled color box image to {out_path}")
    return img

def split_multi_digit_detection(det, avg_digit_width, width_thresh=2):
    text = det['text']
    bbox = det['bbox']
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_width = x_max - x_min
    per_digit_width = box_width / len(text)
    N = len(text)
    if N > 1 and text.isdigit() and per_digit_width > avg_digit_width * width_thresh:
        new_dets = []
        for i, char in enumerate(text):
            if N == 1:
                center_x = (x_min + x_max) // 2
            else:
                center_x = int(x_min + i * (box_width) / (N - 1))
            char_x1 = int(center_x - avg_digit_width // 2)
            char_x2 = int(center_x + avg_digit_width // 2)
            # Clamp to box
            char_x1 = max(x_min, char_x1)
            char_x2 = min(x_max, char_x2)
            char_bbox = [
                (char_x1, y_min),
                (char_x2, y_min),
                (char_x2, y_max),
                (char_x1, y_max)
            ]
            new_dets.append({'text': char, 'bbox': char_bbox})
        return new_dets
    else:
        return [det]

def get_average_colors(image, color_boxes):
    """
    Returns a list of dicts: [{'ph_text': ..., 'avg_color': [B, G, R], 'rect': (x1, y1, x2, y2)}, ...]
    """
    results = []
    for box in color_boxes:
        x1, y1, x2, y2 = box['rect']
        # Clamp to image bounds
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
        roi = image[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            avg_color = [0, 0, 0]
        else:
            avg_color = roi.mean(axis=(0, 1)).tolist()  # BGR order
        results.append({
            'ph_text': box['ph_text'],
            'avg_color': avg_color,
            'rect': (x1, y1, x2, y2)
        })
    return results

def find_closest_color(target_bgr, reference_dict):
    """
    Returns the label and distance of the closest reference color to target_bgr.
    reference_dict: {label: [B, G, R], ...}
    """
    min_dist = float('inf')
    best_label = None
    for label, ref_bgr in reference_dict.items():
        dist = sum((a - b) ** 2 for a, b in zip(target_bgr, ref_bgr)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_label = label
    return best_label, min_dist

def get_average_color_of_box(image, x, y, width, height):
    """
    Returns the average BGR color of the box and the region of interest.
    """
    x1, y1, x2, y2 = int(x), int(y), int(x+width), int(y+height)
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    roi = image[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        avg_color_bgr = [0, 0, 0]
    else:
        avg_color_bgr = [int(round(c)) for c in roi.mean(axis=(0, 1))]
    return avg_color_bgr, (x1c, y1c, x2c, y2c), roi

def highlight_and_label_box(image, box, label, out_path, color_boxes=None, detections=None):
    img = image.copy()
    # Optionally draw all color boxes and/or text boxes
    if color_boxes is not None and detections is not None:
        img = draw_color_boxes(img, detections, color_boxes, out_path=None)
    elif color_boxes is not None:
        img = draw_color_boxes(img, [], color_boxes, out_path=None)
    # Draw the highlighted box and label
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 10
    cv2.putText(img, f"pH {label}", (int(x1), int(y1-20)), font, font_scale, (0, 255, 255), thickness)  # White label, size 3
    if out_path:
        cv2.imwrite(out_path, img)
        print(f"Saved highlighted box with label to {out_path}")
    return img

def find_closest_detected_ph(target_bgr, avg_colors):
    """
    Returns the ph_text and distance of the closest detected color box in avg_colors.
    """
    min_dist = float('inf')
    best_ph = None
    for entry in avg_colors:
        bgr = [int(round(c)) for c in entry['avg_color']]
        dist = sum((a - b) ** 2 for a, b in zip(target_bgr, bgr)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_ph = entry['ph_text']
    return best_ph, min_dist

def ph_from_image(image_path):
    image, detections = detect_text_boxes(image_path)
    # Compute average width of single-digit boxes
    single_digit_widths = [
        max([p[0] for p in det['bbox']]) - min([p[0] for p in det['bbox']])
        for det in detections if len(det['text']) == 1 and det['text'].isdigit()
    ]
    avg_digit_width = np.mean(single_digit_widths) if single_digit_widths else 40
    # Split multi-digit detections
    split_detections = []
    for det in detections:
        split_detections.extend(split_multi_digit_detection(det, avg_digit_width))
    detections = split_detections
    out_path1 = os.path.join(OUTPUT_DIR, "step1_text_boxes.png")
    draw_text_boxes(image, detections, out_path1)

    # --- STEP 2 ---
    rows = cluster_rows(detections)
    color_boxes = define_color_boxes(rows)
    out_path2 = os.path.join(OUTPUT_DIR, "step2_color_boxes.png")
    draw_color_boxes(image, detections, color_boxes, out_path2)

    # --- STEP 3: Get average colors ---
    avg_colors = get_average_colors(image, color_boxes)
    # --- STEP 4: Demo for a given box ---
    avg_bgr, box_coords, roi = get_average_color_of_box(image, 750, 1100, 150, 300) # this is the box of pH strip
    closest_ph, dist = find_closest_detected_ph(avg_bgr, avg_colors)
    highlight_and_label_box(image, box_coords, closest_ph, os.path.join(OUTPUT_DIR, "step3_highlighted_box.png"), color_boxes=color_boxes, detections=detections)
    if closest_ph is not None:
        return closest_ph
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Detect pH from color grid image.")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    args = parser.parse_args()
    result = ph_from_image(args.image_path)
    print(result if result is not None else "NULL")

if __name__ == "__main__":
    main() 