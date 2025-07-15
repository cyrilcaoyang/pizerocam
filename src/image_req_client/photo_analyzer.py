"""
Photo analyzer that give pH readings.
1. Local method: We use Tesseract for OCR
2. Cloud method: We use google cloud vision API for OCR

Author: Yang Cao, Acceleration Consortium
Email: yangcyril.cao@utoronto.ca
Version: 0.2

For Windows computers, please install tesseract first  https://github.com/UB-Mannheim/tesseract/wiki

Instructions to set up google cloud vision:
First install google cloud sdk and restart terminal
- Step 1: Create a Google Cloud Project.
            Go to the Google Cloud Console.
            Click Select a Project > New Project.
            Name your project "pH-OCR" and click Create.
- Step 2: Enable the Vision API.
            In the Cloud Console, navigate to APIs & Services > Library.
            Search for Cloud Vision API and click Enable.
- Step 3: Create Service Account Credentials.
            Go to APIs & Services > Credentials.
            Click Create Credentials > Service Account.
            Name the service account (pH-OCR-Service-Account) and click Create.
            Assign the role Project > Owner (for testing) and click Done.
            Under Service Accounts, click the account you just created.
            Go to the Keys tab > Add Key > Create New Key > JSON.
            Download the JSON key file (e.g., ph-ocr-key.json) and store it securely.
- Step 4: Set Environment Variable
            Set the path to your credentials file in your terminal:
                export GOOGLE_APPLICATION_CREDENTIALS="path/to/ph-ocr-key.json"
            Add this line to your .bashrc/.zshrc to make it permanent.
"""

import os
import sys
import cv2
import re
import numpy as np
import pytesseract
import logging
from pprint import pprint
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from google.cloud import vision
from typing import Tuple, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cropping parameters. Image will be cropped first TODO: add soft cropping with edge detection
startY, endY, startX, endX = 600, 2000, 1000, 4400
downsize_factor = 2         # This factor might have a strong impact on the outcome of Tesseract OCR
width = (endX - startX) // downsize_factor
height = (endY - startY) // downsize_factor

class PhotoAnalyzer:
    """Base class with customizable logger name"""
    LOGGER_NAME = "PhotoAnalyzer"                               # Default base name

    def __init__(self, logger=None):
        self.logger = logger or self._create_class_logger()     # Initialize with existing logger or create default

        self.position_records = defaultdict(list)               # Track positions of numbers in OCR

        # Get settings from environment variables
        self.expected_rows = int(os.getenv('EXPECTED_ROWS', 2))
        self.max_y_variance = int(os.getenv('MAX_Y_VAR', 15))
        self.max_x_variance = int(os.getenv('MAX_X_VAR', 30))
        self.min_row_density = float(os.getenv('MIN_ROW_DENSITY', 0.7))
        
        ph_whitelist_str = os.getenv('PH_WHITELIST', "")
        self.number_whitelist = set(ph_whitelist_str.split(',')) if ph_whitelist_str else set()
        
        self.logger.info("The whitelist is loaded as:")
        self.logger.info(f"{self.number_whitelist}")

    @classmethod
    def _create_class_logger(cls):
        """Create logger using class-specific name"""
        logger = logging.getLogger(f"{cls.LOGGER_NAME}.{cls.__name__}")
        cls._configure_logger(logger)
        return logger

    @staticmethod
    def _configure_logger(logger):
        """Shared logger configuration"""
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)         # Also output the stout by streaming the output
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

    def load_image(
            self,
            img_path
    ) -> np.ndarray:
        """Load the image file as color image"""
        try:
            image = cv2.imread(img_path)
            self.logger.info(f"Image loaded from {img_path}.\nImage dimensions: {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f"Error: {e}.")
            raise ValueError

    def view_image(self, file: np.ndarray):
        self.logger.info("External window opened to view the image.")
        cv2.imshow('Image', file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.logger.info("External window closed.")

    @staticmethod
    def crop_image(file: np.ndarray):
        crop_box = file[startY:endY, startX:endX]           # Define the cropping coordinates
        crop_img = cv2.resize(crop_box, (width, height), interpolation=cv2.INTER_AREA)
        return crop_img

    @staticmethod
    def enhance_image(
            crop_img: np.ndarray,
            contrast: float,
            brightness: int
    ):
        # Enhance the grey image for Tesseract
        adjusted = cv2.convertScaleAbs(crop_img, alpha=float(contrast), beta=float(brightness))
        return adjusted

    def save_image(self, file: np.ndarray, filename: str, directory: os.path):
        file_path = os.path.join(directory, filename)
        cv2.imwrite(str(file_path), file)
        self.logger.info(f"Image saved as {file_path}.")

    def capture_colors(
            self, image:
            np.ndarray,
            detections: list
    ):
        """
        Polygon-aware color capture for both OCR implementations
        Returns image with ROIs marked and updated detections with colors
        """
        if not detections:
            self.logger.info('No detections for color capture')
            return image, []

        dimensions = []                                         # Calculate average dimensions from polygon vertices
        for det in detections:
            x_coords = [p[0] for p in det['bounding_box']]
            y_coords = [p[1] for p in det['bounding_box']]
            dimensions.append((
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords)
            ))
        avg_width = int(np.mean([w for w, h in dimensions]))
        avg_height = int(np.mean([h for w, h in dimensions]))

        marked = image.copy()                                       # Mark numbers recognized
        for det in detections:
            x_coords = [p[0] for p in det['bounding_box']]          # Get bounding box extremes
            y_coords = [p[1] for p in det['bounding_box']]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            roi_y1 = y_max + int(0.5 * avg_height)                  # Calculate color ROI (2x avg height below text)
            roi_y2 = roi_y1 + int(1.5 * avg_height)
            roi_x1 = x_min
            roi_x2 = x_max

            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]               # Get color from ROI
            det['color'] = np.mean(roi, axis=(0, 1)).tolist() if roi.size > 0 else [0, 0, 0]
            cv2.rectangle(marked, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)           # Draw visualization
            cv2.polylines(marked, [np.array(det['bounding_box'])], True, (0, 255, 255), 2)
        return marked, detections

    @staticmethod
    def _base_normalization(text: str) -> str:
        """Universal normalization for all OCR implementations"""
        clean = re.sub(r'[^\d.,]', '', text)            # Strip non-numeric characters except . and ,
        normalized = clean.replace(',', '.').lstrip('0')# Handle European decimals and leading zeros

        parts = normalized.split('.')                               # Split into components
        if len(parts) == 1:
            return parts[0] or '0'                                  # Handle empty string case
        integer_part = parts[0] or '0'
        decimal_part = parts[1][:1]                                 # Take only first decimal digit
        return f"{integer_part}.{decimal_part}"

    def diff_image(
            self,
            img_path_dry: str,
            img_path_wet: str
    ):
        """
        Takes the image before and after a drop of analyte solution is dropped.
        Use contract to find the ROI
        :returns:
            diff_map: path to generated heatmap image,
            coordinates: tuple of (x, y, diameter) or None
        """
        img_dry = self.crop_image(cv2.imread(img_path_dry))         # Load and validate images
        img_wet = self.crop_image(cv2.imread(img_path_wet))

        if img_dry is None or img_wet is None:
            self.logger.error("Failed to load one or both images")
            return None, None
        if img_dry.shape != img_wet.shape:
            self.logger.error("Image dimensions mismatch")
            return None, None

        lab_dry = cv2.cvtColor(img_dry, cv2.COLOR_BGR2LAB)          # Calculate perceptual color difference
        lab_wet = cv2.cvtColor(img_wet, cv2.COLOR_BGR2LAB)

        delta_e = np.sqrt(                                          # Generate differential image
            np.square(lab_dry[:, :, 1].astype(np.float32) - lab_wet[:, :, 1]) +
            np.square(lab_dry[:, :, 2].astype(np.float32) - lab_wet[:, :, 2])
        )
        norm_diff = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        diff_map = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)

        blurred = cv2.GaussianBlur(delta_e, (51, 51), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)             # Analyze change region

        if max_val < 5:
            self.logger.info("No significant changes detected")
            return diff_map, None

        y_grid, x_grid = np.indices(delta_e.shape)                  # Calculate weighted centroid
        weights = np.maximum(blurred - max_val * 0.5, 0)
        total_weight = np.sum(weights)

        if total_weight < 1e-6:
            return diff_map, None
        center_x = np.sum(x_grid * weights) / total_weight
        center_y = np.sum(y_grid * weights) / total_weight

        x_dev = np.sqrt(np.sum(weights * (x_grid - center_x) ** 2) / total_weight)
        y_dev = np.sqrt(np.sum(weights * (y_grid - center_y) ** 2) / total_weight)
        diameter = 2.3548 * np.sqrt(x_dev ** 2 + y_dev ** 2)        # Calculate effective diameter
        return diff_map, (int(center_x), int(center_y), int(diameter))

    def aggregate_positions(self):
        """Simple aggregation of positions by median with counts"""
        aggregated = {}
        for num, polygons in self.position_records.items():
            try:
                bboxes = []                                         # Convert polygons to bounding boxes
                for poly in polygons:
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    bboxes.append([
                        min(x_coords),  # x_min
                        min(y_coords),  # y_min
                        max(x_coords),  # x_max
                        max(y_coords)  # y_max
                    ])

                median_box = np.median(bboxes, axis=0).astype(int)  # Calculate median position
                aggregated[num] = {
                    'bbox': median_box,
                    'count': len(bboxes),
                    'all_boxes': bboxes                             # For outlier removal
                }
            except Exception as e:
                self.logger.error(f"Aggregation failed for {num}: {str(e)}")
        return aggregated

    def remove_outliers(self, aggregated_data, max_deviation=50):
        """Remove positions deviating too far from median"""
        filtered = {}
        for num, data in aggregated_data.items():
            try:
                median = data['bbox']
                boxes = np.array(data['all_boxes'])
                distances = np.abs(boxes - median)

                outlier_mask = (                                     # Create outlier mask
                        (distances[:, 0] < max_deviation) &  # x_min
                        (distances[:, 1] < max_deviation) &  # y_min
                        (distances[:, 2] < max_deviation) &  # x_max
                        (distances[:, 3] < max_deviation)  # y_max
                )

                filtered_boxes = boxes[outlier_mask]
                if filtered_boxes.size == 0:
                    continue
                filtered[num] = {
                    'bbox': np.median(filtered_boxes, axis=0).astype(int),
                    'count': len(filtered_boxes)
                }
            except Exception as e:
                self.logger.error(f"Outlier removal failed for {num}: {str(e)}")
        return filtered

    def label_image(self, crop_image, filtered_data):
        """Draw labels with count and color patches"""
        labeled = crop_image.copy()

        for num, data in filtered_data.items():
            x_min, y_min, x_max, y_max = data['bbox']
            count = data['count']

            data['color'] = self._get_reference_color(data['bbox'], crop_image)
            data['read_pos'] = {
                'x': x_min,
                'y': int(2 * y_max - y_min),
                'width': x_max - x_min,
                'height': int((y_max - y_min) * 2)
            }

            text = f"{num} ({count})"                               # Draw number and count
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            text_x = max(x_min, 0)
            text_y = max(y_min - th // 2, th)
            cv2.putText(labeled, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            avg_color = self._get_reference_color(data['bbox'], crop_image)
            box_height = y_max - y_min
            color_y1 = y_max + box_height
            color_y2 = color_y1 + int(box_height * 2)

            cv2.rectangle(labeled,
                          (x_min, color_y1),
                          (x_max, color_y2),
                          (0, 255, 0), 2)  # 2 pixels thick, no fill
        return labeled

    def read_ph_from_diff(
            self,
            image_path: str,
            diff_coordinates: Tuple[int, int, int],  # (center_x, center_y, diameter)
            aggregated_nums: Dict[str, Dict]
    ):
        """
        Reads pH value from automatically detected region of interest
        :param image_path: Path to the input image
        :param diff_coordinates: Tuple from diff_image (center_x, center_y, diameter)
        :param aggregated_nums: Filtered data from remove_outliers()
        :returns: Tuple (estimated pH value, annotated image)
        """
        try:
            # Load and prepare image
            image = self.load_image(image_path)
            crop = self.crop_image(image)
            directory, file_name = os.path.split(image_path)
            file_name_no_extension = file_name[:-4]

            center_x, center_y, diameter = diff_coordinates         # Calculate ROI from diff coordinates
            radius = diameter // 4                                  # Use full detection diameter for reading
            annotated = crop.copy()                                 # Create annotated image

            cv2.circle(annotated,                                   # Draw measurement circle (thicker yellow border)
                       (center_x, center_y), radius,
                       (0, 255, 255),  # Yellow color
                       3)  # Thicker border

            # Extract ROI color from circular area
            y, x = np.ogrid[:crop.shape[0], :crop.shape[1]]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            read_roi = crop[mask]

            if read_roi.size == 0:
                raise ValueError("Empty circular ROI")

            roi_color = np.mean(read_roi, axis=0)
            self.logger.info(f'ROI Color (BGR): {roi_color.astype(int)}')

            # Find closest pH match
            color_distances = {
                ph: np.linalg.norm(roi_color - self._get_reference_color(data['bbox'], crop))
                for ph, data in aggregated_nums.items()
            }

            if not color_distances:
                raise ValueError("No reference colors available")

            closest_ph = min(color_distances.items(), key=lambda x: x[1])[0]

            # Draw pH text at circle top
            text_pos = (center_x - 20, center_y - radius - 10)
            cv2.putText(annotated, f"pH: {closest_ph}",
                        text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)  # Yellow text
            return closest_ph, annotated

        except Exception as e:
            self.logger.error(f"pH reading failed: {str(e)}")
        return str(e), crop if 'crop' in locals() else None

    @staticmethod
    def _get_reference_color(bbox, crop_image):
        """Helper to get color from reference number area"""
        x_min, y_min, x_max, y_max = bbox
        box_height = y_max - y_min
        color_y1 = y_max + box_height
        color_y2 = min(y_max + 3 * box_height, crop_image.shape[0])

        if color_y1 >= color_y2:
            return np.array([0, 0, 0])

        color_roi = crop_image[color_y1:color_y2, x_min:x_max]
        return np.mean(color_roi, axis=(0, 1)) if color_roi.size > 0 else np.array([0, 0, 0])

class TesseractAnalyzer(PhotoAnalyzer):
    LOGGER_NAME = "OCR.Tesseract"
    def __init__(self):
        super().__init__()
        with open(Path(__file__).resolve().parent / 'image_req_client_settings.yaml', 'r') as file:
            data = yaml.safe_load(file)
        self.path_tesseract = data['Path_Tesseract']
        pytesseract.pytesseract.tesseract_cmd = self.path_tesseract

    def text_detection(self,processed_image: np.ndarray):
        """
        Tesseract-specific text detection with pH paper optimizations
        """
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        detection_data = pytesseract.image_to_data(
            processed_image,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        results = []                                       # Init a dict to store the locations of the numbers
        for i in range(len(detection_data['text'])):
            raw_text = detection_data['text'][i].strip()
            if not raw_text:
                continue                                            # Skip empty or non-numeric text
            normalized = self._base_normalization(raw_text)         # Universal normalization

            if normalized in self.number_whitelist:                 # Store only if in whitelist
                left = detection_data['left'][i]
                top = detection_data['top'][i]
                width = detection_data['width'][i]
                height = detection_data['height'][i]

                results.append({
                    'ph_value': normalized,
                    'bounding_box': [
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height)
                    ],
                    'raw_text': raw_text                            # For debugging OCR errors
                })
        return results

    def find_num_loc_colors(
            self,
            image: np.ndarray,
            contrast: float,
            brightness: int
    ):
        crop = self.crop_image(image)
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)               # Convert image to greyscale for OCR
        enhanced_grey = self.enhance_image(grey, contrast, brightness)
        denoised_grey = cv2.GaussianBlur(enhanced_grey, (5, 5), 0)

        num_local = self.text_detection(denoised_grey)              # OCR w/ tesseract, num_local has no color info
        if num_local:                                               # Update num_local to include color info
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else:
            marked_crop = crop
        return marked_crop, denoised_grey, num_local

    def tesseract_ocr_optimization_flow(
            self,
            color_image: np.ndarray,
            step: int = 25,
            save_iterations: bool = False,
            output_dir: Path = None
    ) -> Tuple[Dict, np.ndarray, np.ndarray, plt.Figure]:
        """
        Optimize OCR parameters through contrast/brightness grid search
        :returns:
            filtered: The filtered information of known pH value and colors from the same image
            labeled_image: np.ndarray, the cropped color image with pH numbers marked on them
            ocr_viz: np.ndarray, visualization of the OCR mapped on cropped image
            opt_plot: Image object (plot can be saved with matplotlib)
        """
        self.logger.info("Starting Tesseract OCR optimization flow")
        self.position_records = defaultdict(list)

        crop = self.crop_image(color_image)
        detection_viz = np.zeros(crop.shape[:2], dtype=np.uint16)
        brightness_steps = range(-200, 201, step)
        contrast_steps = [i / 100 for i in range(0, 301, step)]
        opt_results = np.zeros((len(brightness_steps), len(contrast_steps)))
        max_num = 0

        detections_dir = None
        if save_iterations and output_dir:                              # For debugging, save the iteration images>0 OCRs
            detections_dir = output_dir / "ocr_detections"
            detections_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving iteration images to {detections_dir}")

        for j_idx, brightness in enumerate(brightness_steps):
            for i_idx, contrast in enumerate(contrast_steps):
                try:
                    marked_crop, _, detections = self.find_num_loc_colors(
                        color_image, contrast=contrast, brightness=brightness
                    )

                    current_count = len(detections)                     # Add detection results to positional records
                    opt_results[j_idx, i_idx] = current_count
                    for detection in detections:
                        ph_value = detection['ph_value']
                        polygon = detection['bounding_box']
                        self.position_records[ph_value].append(polygon)

                    self._update_detection_matrix(detections, detection_viz)
                    self.logger.info(
                        f"Trying contrast={contrast}, brightness={brightness} found {current_count} numbers."
                    )

                    if save_iterations and current_count > 0 and detections_dir:
                        iter_filename = (f"contrast_{contrast/100:.2f}_brightness_{brightness}.jpg")
                        output_path = detections_dir / iter_filename
                        cv2.imwrite(str(output_path), marked_crop)
                        self.logger.debug(f"Saved iteration: {output_path}")
                    if current_count > max_num:
                        max_num = current_count

                except Exception as e:
                    self.logger.error(f"Error processing {contrast=:.2f}, {brightness=}: {str(e)}")

        aggregated = self.aggregate_positions()
        filtered = self.remove_outliers(aggregated)
        labeled_image = self.label_image(crop, filtered)
        ocr_viz = self._create_ocr_visualization(detection_viz)
        opt_plot = self._create_optimization_plot(opt_results, brightness_steps, contrast_steps)

        self.logger.info(f"Optimization complete. Max detections: {max_num}")
        return filtered, labeled_image, ocr_viz, opt_plot

    @staticmethod
    def _update_detection_matrix(
            detections: List[Dict],
            matrix: np.ndarray
    ) -> None:
        """Update detection frequency matrix"""
        for entry in detections:
            x0, y0 = entry['bounding_box'][0]
            x1 = entry['bounding_box'][1][0]
            y2 = entry['bounding_box'][2][1]
            matrix[y0:y2, x0:x1] += 1

    @staticmethod
    def _create_ocr_visualization(detection_matrix: np.ndarray) -> np.ndarray:
        """Generate OCR detection visualization"""
        normalized = 255 - (detection_matrix / np.max(detection_matrix) * 255).astype(np.uint8) \
            if np.max(detection_matrix) > 0 else np.full_like(detection_matrix, 255, np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _create_optimization_plot(
            opt_results: np.ndarray,
            brightness_steps: range,
            contrast_steps: List[float]
    ) -> plt.Figure:
        """Generate parameter optimization heatmap plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(opt_results, aspect='auto',
                       extent=(contrast_steps[0], contrast_steps[-1],
                               brightness_steps[0], brightness_steps[-1]),
                       origin='lower')

        plt.colorbar(im, ax=ax, label='Numbers Recognized')
        ax.set(
            title='OCR Parameter Optimization',
            xlabel='Contrast (1.0-3.0)',
            ylabel='Brightness (-200-200)'
        )
        plt.close(fig)  # Prevent duplicate displays
        return fig

class CloudVisionAnalyzer(PhotoAnalyzer):
    LOGGER_NAME = "OCR.CloudVision"

    def __init__(self):
        super().__init__()
        self.client = vision.ImageAnnotatorClient()                 # Setting up google cloud API

    def text_detection(self, color_image: np.ndarray):
        """
        Use Google Cloud Vision API to detect text in the image
        """
        _, encoded_image = cv2.imencode(".jpg", color_image)
        image_obj = vision.Image(content=encoded_image.tobytes())   # Convert np.ndarry to Image Object
        response = self.client.document_text_detection(image=image_obj)

        results = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        raw_text = ''.join([symbol.text for symbol in word.symbols])

                        normalized = self._base_normalization(raw_text) # Universal normalization

                        vertices = [(vertex.x, vertex.y)        # Get polygon vertices
                                    for vertex in word.bounding_box.vertices]
                        results.append({
                            'ph_value': normalized,
                            'bounding_box': vertices,
                            'raw_text': raw_text                # For debugging
                        })
        if response.error.message:
            raise Exception(f'{response.error.message}')
        return results

    def find_num_loc_colors(
            self,
            image: np.ndarray
    ):
        """
        Get the locations and colors of numbers using Google Cloud Vision API
        """
        crop = self.crop_image(image)
        num_local = self.text_detection(crop)

        if num_local:
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else:
            marked_crop = crop
        return marked_crop, crop, num_local

    def google_cloud_ocr_flow(
            self,
            image: np.ndarray,
            output_dir: Path = None
    ) -> Tuple[Dict, np.ndarray, np.ndarray, plt.Figure]:
        """
        Google Cloud OCR processing flow without parameter optimization
        :returns:
            filtered: Filtered data of pH values and positions
            labeled_image: Image with pH numbers marked
            ocr_viz: Visualization of OCR detections
            opt_plot: Placeholder figure (no optimization performed)
        """
        self.logger.info("Starting Google Cloud OCR flow")
        self.position_records = defaultdict(list)  # Reset for each run

        # Process image once
        marked_crop, crop, detections = self.find_num_loc_colors(image)
        pprint(detections)

        # Update position records with detected bounding boxes
        for detection in detections:
            ph_value = detection['ph_value']
            polygon = detection['bounding_box']
            self.position_records[ph_value].append(polygon)

        # Create detection matrix for visualization
        detection_viz = np.zeros(crop.shape[:2], dtype=np.uint16)
        self._update_detection_matrix(detections, detection_viz)

        # Aggregate and filter positions
        aggregated = self.aggregate_positions()

        # Label the image with numbers and colors
        labeled_image = self.label_image(crop, aggregated)

        # Generate OCR visualization
        ocr_viz = self._create_ocr_visualization(detection_viz)

        # Dummy optimization plot (no optimization performed)
        fig = plt.figure()
        plt.close(fig)  # Close to prevent display

        self.logger.info("Google Cloud OCR flow completed successfully.")
        return aggregated, labeled_image, ocr_viz, fig

    @staticmethod
    def _update_detection_matrix(
            detections: List[Dict],
            matrix: np.ndarray
    ) -> None:
        """Update detection frequency matrix for visualization"""
        for entry in detections:
            x0, y0 = entry['bounding_box'][0]
            x1 = entry['bounding_box'][1][0]
            y2 = entry['bounding_box'][2][1]
            matrix[y0:y2, x0:x1] += 1

    @staticmethod
    def _create_ocr_visualization(detection_matrix: np.ndarray) -> np.ndarray:
        """Generate OCR detection visualization heatmap"""
        if np.max(detection_matrix) > 0:
            normalized = 255 - (detection_matrix / np.max(detection_matrix) * 255).astype(np.uint8)
        else:
            normalized = np.full_like(detection_matrix, 255, dtype=np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":

    # # Load images
    # path_to_image_dry = ("photos/Simple_workflow/ph_10/capture_20250226-135950_200200200.jpg")
    # path_to_image_wet = ("photos/Simple_workflow/ph_10/capture_20250226-140013_200200200.jpg")

    # Load images
    path_to_image_dry = ("photos/capture_20250228-143619_255255255.jpg")
    path_to_image_wet = ("photos/capture_20250228-143603_255255255.jpg")
    color_image = cv2.imread(path_to_image_wet)
    output_dir = Path(path_to_image_wet).parent

    if not os.path.exists(path_to_image_wet):
        raise FileNotFoundError(f"Image not found: {path_to_image_wet}")
    directory, file_name = os.path.split(path_to_image_wet)
    file_name_no_extension = file_name[:-4]

    ## USE CASE 1: USE TESSERACT OCR
    tesseract_analyzer = TesseractAnalyzer()
    logging.basicConfig(level=logging.INFO)

    # Macbook setting override, comment off for Windows machines
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

    # Find ROI using contrast of wet and dry images
    differential_image, ROI = tesseract_analyzer.diff_image(path_to_image_dry, path_to_image_wet)
    if differential_image is not None:
        tesseract_analyzer.save_image(  # Save labelled crop
            differential_image, f"tess_{file_name_no_extension}_diff_image.png", directory)
    tesseract_analyzer.logger.info(f"ROI_X = {ROI[0]}, ROI_Y = {ROI[1]}, diameter = {ROI[2]}")

    steps = [50]
    for step in steps:
        try:
            filtered_data, labeled_image, ocr_viz, opt_heat_map = (
                tesseract_analyzer.tesseract_ocr_optimization_flow(
                    color_image, step=step, save_iterations=True, output_dir=output_dir
                )
            )
            base_name = f"tess_{file_name_no_extension}_step{step}"      # Save all outputs to the original image directory
            tesseract_analyzer.save_image(                          # Save labeled image
                labeled_image,f"{base_name}_labeled.png", directory
            )
            tesseract_analyzer.save_image(                          # Save OCR visualization
                ocr_viz,f"{base_name}_detections.png", directory
            )

            heatmap_path = Path(directory) / f"{base_name}_opt_heatmap.png"     # Save optimization heatmap plot
            opt_heat_map.savefig(str(heatmap_path),dpi=300,bbox_inches='tight',facecolor='white')

            # Save numerical results
            results_path = Path(directory) / f"{base_name}_results.txt"
            with open(results_path, 'w') as f:
                pprint(filtered_data, stream=f)

            # pH reading using diff_image coordinates
            if filtered_data and ROI is not None:
                ph_value, ph_image = tesseract_analyzer.read_ph_from_diff(
                    path_to_image_wet,
                    ROI,
                    filtered_data
                )
                tesseract_analyzer.logger.info(f"Final pH measurement: {ph_value}")
                tesseract_analyzer.save_image(
                    ph_image,
                    f"{base_name}_ph_result.png",
                    directory
                )
            else:
                tesseract_analyzer.logger.error("No valid data for pH calculation")

        except Exception as e:
            tesseract_analyzer.logger.error(f"Analysis failed: {str(e)}")


    ## USE CASE 2: USE GOOGLE CLOUD VISION OCR
    cloud_analyzer = CloudVisionAnalyzer()
    logging.basicConfig(level=logging.INFO)

    # Find ROI using same dry/wet contrast method
    differential_image, ROI = cloud_analyzer.diff_image(path_to_image_dry, path_to_image_wet)
    if differential_image is not None:
        cloud_analyzer.save_image(
            differential_image, f"cloud_{file_name_no_extension}_diff_image.png", directory)
    cloud_analyzer.logger.info(f"CLOUD ROI - X: {ROI[0]}, Y: {ROI[1]}, Diameter: {ROI[2]}")

    try:
        filtered_data, labeled_image, ocr_viz, _ = cloud_analyzer.google_cloud_ocr_flow(
            color_image, output_dir=output_dir
        )

        base_name = f"cloud_{file_name_no_extension}"
        cloud_analyzer.save_image(  # Save labeled image with color patches
            labeled_image, f"{base_name}_labeled.png", directory
        )
        cloud_analyzer.save_image(  # Save detection frequency visualization
            ocr_viz, f"{base_name}_detections.png", directory
        )

        # Save numerical results
        results_path = Path(directory) / f"{base_name}_results.txt"
        with open(results_path, 'w') as f:
            pprint(filtered_data, stream=f)

        # pH reading using same coordinates from diff_image
        if filtered_data and ROI is not None:
            ph_value, ph_image = cloud_analyzer.read_ph_from_diff(
                path_to_image_wet,
                ROI,
                filtered_data
            )
            cloud_analyzer.logger.info(f"Cloud Vision pH: {ph_value}")
            cloud_analyzer.save_image(
                ph_image,
                f"{base_name}_ph_result.png",
                directory
            )
        else:
            cloud_analyzer.logger.error("No valid data for pH calculation")

    except Exception as e:
        cloud_analyzer.logger.error(f"Cloud Vision analysis failed: {str(e)}")