import os
import cv2
import numpy as np
from pathlib import Path
from google.cloud import vision
import itertools
import argparse
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Google Cloud credentials if provided in .env
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# --- CONFIG ---
# Path to the image to analyze
IMAGE_PATH = "C:/Users/yasee/Documents/Projects/pizerocam/test_images/capture_20250909-171227_200200200.jpg"
  # Change to your image path

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

def cluster_rows(detections, row_gap_thresh=None):
    """Cluster text boxes into rows based on y-coordinate proximity using text height as tolerance."""
    if not detections:
        return []
    
    # Calculate average text height to use as clustering tolerance
    text_heights = []
    for det in detections:
        bbox = det['bbox']
        text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
        text_heights.append(text_height)
        # Also calculate centers for later use
        det['y_center'] = int(np.mean([p[1] for p in bbox]))
        det['x_center'] = int(np.mean([p[0] for p in bbox]))
    
    # Use average text height as tolerance, with fallback
    if row_gap_thresh is None:
        avg_text_height = np.mean(text_heights) if text_heights else 40
        row_gap_thresh = int(avg_text_height)
        print(f"Using text-based row clustering tolerance: {row_gap_thresh} pixels (avg text height: {avg_text_height:.1f})")
    
    # Sort by y_center
    detections_sorted = sorted(detections, key=lambda d: d['y_center'])
    
    # Group into rows using the text height-based tolerance
    rows = []
    for k, group in itertools.groupby(detections_sorted, key=lambda d: d['y_center']//row_gap_thresh):
        rows.append(list(group))
    
    print(f"Clustered {len(detections)} detections into {len(rows)} rows")
    return rows

def define_color_boxes(rows, delta_y_frac=0.4, width_frac=0.5):
    """For each number, define a color box below it."""
    # Calculate text box heights to ensure minimum color box height
    all_detections = [det for row in rows for det in row]
    text_heights = []
    for det in all_detections:
        bbox = det['bbox']
        text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
        text_heights.append(text_height)
    
    if text_heights:
        avg_text_height = np.mean(text_heights)
        min_color_box_height = int(avg_text_height)  # Minimum height = text height
        print(f"Average text height: {avg_text_height:.1f} pixels, minimum color box height: {min_color_box_height}")
    else:
        min_color_box_height = 20  # fallback minimum
        avg_text_height = 20
    
    if len(rows) < 2:
        print("Warning: Less than 2 rows detected. Using text box height to estimate color box size.")
        # Use text height as a reasonable scale reference
        row_delta = avg_text_height * 3  # Color box should be ~3x text height below
        print(f"Estimated row_delta from text height: {row_delta:.1f} pixels")
    else:
        # Use mean y distance between row centers
        row_delta = abs(np.mean([det['y_center'] for det in rows[1]]) - np.mean([det['y_center'] for det in rows[0]]))
        print(f"Calculated row_delta from row spacing: {row_delta:.1f} pixels")
    
    # Add minimum spacing to ensure color boxes don't overlap with text
    min_spacing = avg_text_height * 0.5  # At least half text height spacing
    print(f"Using minimum spacing: {min_spacing:.1f} pixels")
    
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
            
            # Calculate this specific text box height and width
            text_height = max([p[1] for p in bbox]) - min([p[1] for p in bbox])
            text_width = max([p[0] for p in bbox]) - min([p[0] for p in bbox])
            
            # Calculate width ensuring it's at least as wide as the text
            calculated_width = int(avg_x_dist * width_frac)
            width = max(calculated_width, text_width, 100)  # Minimum 100 pixels wide
            
            calculated_height = int(row_delta * delta_y_frac)
            
            # Ensure height is at least the height of this number's bounding box
            height = max(calculated_height, text_height, min_color_box_height)
            
            x1 = int(x_center - width//2)
            x2 = int(x_center + width//2)
            
            # FIXED: More robust positioning - ensure adequate spacing below text
            spacing = max(min_spacing, row_delta * 0.2)  # Use larger of minimum spacing or 30% of row_delta
            y1 = int(y_max + spacing)
            y2 = int(y1 + height)
            
            print(f"pH {det['text']}: text_height={text_height}, spacing={spacing:.1f}, box=({x1},{y1},{x2},{y2})")
            
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
        cv2.putText(img, f"pH {box['ph_text']}", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 10)  # White label, size 3
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
    Only includes boxes with valid pH text (numeric, between 0-14).
    """
    results = []
    for box in color_boxes:
        # Filter: only keep valid pH values (numeric strings that can be converted to float 0-14)
        ph_text = box['ph_text']
        try:
            ph_value = float(ph_text)
            if not (0 <= ph_value <= 14):
                print(f"Skipping invalid pH value (out of range): {ph_text}")
                continue
        except ValueError:
            print(f"Skipping non-numeric pH text: {ph_text}")
            continue
        
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

def convert_bgr_to_color_space(bgr_color, color_space='rgb'):
    """
    Convert BGR color to specified color space.
    Args:
        bgr_color: BGR color as [B, G, R] list/array
        color_space: 'rgb', 'lab', or 'hsv'
    Returns:
        Converted color as numpy array
    """
    bgr_array = np.array(bgr_color, dtype=np.uint8).reshape(1, 1, 3)
    
    if color_space.lower() == 'rgb':
        # Convert BGR to RGB for consistency
        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return rgb.flatten().astype(np.float32)
    elif color_space.lower() == 'lab':
        # Convert BGR to LAB
        lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
        return lab.flatten().astype(np.float32)
    elif color_space.lower() == 'hsv':
        # Convert BGR to HSV
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        return hsv.flatten().astype(np.float32)
    else:
        raise ValueError(f"Unsupported color space: {color_space}. Use 'rgb', 'lab', or 'hsv'.")

def find_closest_detected_ph(target_bgr, avg_colors, color_space='rgb'):
    """
    Returns the ph_text and distance of the closest detected color box in avg_colors.
    Args:
        target_bgr: Target color in BGR format [B, G, R]
        avg_colors: List of color data from get_average_colors
        color_space: 'rgb', 'lab', or 'hsv' - color space for distance calculation
    Returns:
        (best_ph, min_dist) tuple
    """
    min_dist = float('inf')
    best_ph = None
    
    # Convert target color to specified color space
    target_converted = convert_bgr_to_color_space(target_bgr, color_space)
    
    for entry in avg_colors:
        bgr = [int(round(c)) for c in entry['avg_color']]
        # Convert reference color to specified color space
        ref_converted = convert_bgr_to_color_space(bgr, color_space)
        # Calculate Euclidean distance in the color space
        dist = np.linalg.norm(ref_converted - target_converted)
        if dist < min_dist:
            min_dist = dist
            best_ph = entry['ph_text']
    return best_ph, min_dist

def interpolate_ph_from_distances(distances):
    """
    Interpolate pH value to one decimal place based on two closest references.
    Args:
        distances: Dict of {pH_string: distance}
    Returns:
        Interpolated pH value rounded to 1 decimal place as float
    """
    # Convert pH strings to floats and sort by distance
    ph_distances = [(float(ph), dist) for ph, dist in distances.items()]
    sorted_phs = sorted(ph_distances, key=lambda x: x[1])
    
    if len(sorted_phs) < 2:
        # Not enough references for interpolation
        return round(sorted_phs[0][0], 1)
    
    # Get two closest pH values
    ph1, dist1 = sorted_phs[0]
    ph2, dist2 = sorted_phs[1]
    
    # If distances are very similar or first distance is zero, return closest
    if dist1 == 0 or dist2 == 0:
        return round(ph1, 1)
    
    # Inverse distance weighting for interpolation
    weight1 = 1.0 / dist1
    weight2 = 1.0 / dist2
    total_weight = weight1 + weight2
    
    interpolated_ph = (ph1 * weight1 + ph2 * weight2) / total_weight
    return round(interpolated_ph, 1)

def ph_from_image(image_path, return_all_color_spaces=False, output_dir=None, interpolate=True):
    """
    Detect pH from color grid image.
    
    Args:
        image_path: Path to the image file to analyze
        return_all_color_spaces: If True, returns dict with results from RGB, LAB, and HSV.
                                 If False, returns single pH value from RGB (default behavior)
        output_dir: Directory to save annotated images. If None, saves to ~/Pictures/pH_photos/
                   If "same", saves to same directory as input image
        interpolate: If True, interpolates pH to 1 decimal place; if False, returns exact match
    
    Returns:
        If return_all_color_spaces=False: pH value as string or float (e.g., "7" or 7.3)
        If return_all_color_spaces=True: dict like {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 
                                                     'distances': {'rgb': 15.2, 'lab': 12.1, 'hsv': 18.5}}
    """
    # Setup output directory
    if output_dir == "same":
        # Save in same directory as input image
        output_dir = Path(image_path).parent
    elif output_dir is None:
        # Default to ~/Pictures/pH_photos/
        home_dir = Path.home()
        output_dir = home_dir / "Pictures" / "pH_photos"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base filename without extension from input image
    input_filename = Path(image_path).stem  # e.g., "capture_20250715-151115_200200200"
    
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
    out_path1 = output_dir / f"{input_filename}_step1_text_boxes.png"
    draw_text_boxes(image, detections, str(out_path1))

    # --- STEP 2 ---
    rows = cluster_rows(detections)
    
    color_boxes = define_color_boxes(rows)
    out_path2 = output_dir / f"{input_filename}_step2_color_boxes.png"
    draw_color_boxes(image, detections, color_boxes, str(out_path2))

    # --- STEP 3: Get average colors ---
    avg_colors = get_average_colors(image, color_boxes)
    # --- STEP 4: Find pH using different color spaces ---
    avg_bgr, box_coords, roi = get_average_color_of_box(image, 1350, 1250, 150, 300) # this is the box of pH strip - adjusted for left side pink strip
    
    if return_all_color_spaces:
        # Calculate pH for all three color spaces
        results = {}
        all_distances = {}
        color_spaces = ['rgb', 'lab', 'hsv']
        
        for color_space in color_spaces:
            # Get distances to all pH values in this color space
            distances_dict = {}
            target_converted = convert_bgr_to_color_space(avg_bgr, color_space)
            
            for entry in avg_colors:
                bgr = [int(round(c)) for c in entry['avg_color']]
                ref_converted = convert_bgr_to_color_space(bgr, color_space)
                dist = np.linalg.norm(ref_converted - target_converted)
                distances_dict[entry['ph_text']] = dist
            
            # Interpolate or get closest
            if interpolate:
                ph_value = interpolate_ph_from_distances(distances_dict)
                results[color_space] = ph_value
            else:
                closest_ph = min(distances_dict, key=distances_dict.get)
                results[color_space] = closest_ph
            
            # Store minimum distance for reference
            all_distances[color_space] = min(distances_dict.values())
            
            print(f"{color_space.upper()} color space: pH={results[color_space]}, min distance={all_distances[color_space]:.2f}")
        
        # Save highlighted image with RGB result (default)
        out_path3 = output_dir / f"{input_filename}_step3_highlighted_box.png"
        highlight_and_label_box(image, box_coords, results['rgb'], str(out_path3), 
                               color_boxes=color_boxes, detections=detections)
        
        results['distances'] = all_distances
        return results
    else:
        # Single color space mode - return pH using RGB
        distances_dict = {}
        target_converted = convert_bgr_to_color_space(avg_bgr, 'rgb')
        
        for entry in avg_colors:
            bgr = [int(round(c)) for c in entry['avg_color']]
            ref_converted = convert_bgr_to_color_space(bgr, 'rgb')
            dist = np.linalg.norm(ref_converted - target_converted)
            distances_dict[entry['ph_text']] = dist
        
        # Interpolate or get closest
        if interpolate:
            ph_result = interpolate_ph_from_distances(distances_dict)
        else:
            ph_result = min(distances_dict, key=distances_dict.get)
        
        out_path3 = output_dir / f"{input_filename}_step3_highlighted_box.png"
        highlight_and_label_box(image, box_coords, ph_result, str(out_path3), 
                               color_boxes=color_boxes, detections=detections)
        
        if ph_result is not None:
            return ph_result
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