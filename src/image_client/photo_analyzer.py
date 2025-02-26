"""
Photo analyzer that give pH readings.

Author: Yang Cao, Acceleration Consortium
Email: yangcyril.cao@utoronto.ca
Version: 0.2

For Windows computers, please install tesseract first  https://github.com/UB-Mannheim/tesseract/wiki

For use of EasyOCR, please first install PyTorch according to: https://pytorch.org/
# Then install: pip3 install easyocr
"""

import os
import cv2
import yaml
import json
import numpy as np
import pytesseract
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sdl_utils import get_logger

# Cropping parameters. Image will be cropped first TODO: add soft cropping with edge detection
startY, endY, startX, endX = 800, 1900, 1000, 4000
downsize_factor = 2         # This factor might have a strong impact on the outcome of OCR
width = (endX - startX) // downsize_factor
height = (endY - startY) // downsize_factor


class PhotoAnalyzer:
    def __init__(self):
        self.logger = self._setup_logger()
        script_dir = Path(__file__).resolve().parent        # Get the directory where this script is located
        self.position_records = defaultdict(list)           # Track positions across conditions

        with open(script_dir / 'image_client_settings.yaml', 'r') as file:
            data = yaml.safe_load(file)
        self.path_tesseract = data['Path_Tesseract']
        pytesseract.pytesseract.tesseract_cmd = self.path_tesseract

        # Load whitelist and tolerance of OCR
        self.number_whitelist = set(map(str, data['PH_Whitelist']))
        self.expected_rows = data.get('Expected_Rows', 2)
        self.max_y_variance = data.get('Max_Y_Var', 15)
        self.max_x_variance = data.get('Max_X_Var', 30)
        self.min_row_density = data.get('Min_Row_Density', 0.7)

    @staticmethod
    def _setup_logger():
        logger = get_logger("PhotoAnalyzerLogger")          # Create the logger and file handler
        return logger

    def load_image(self, img_path):
        """
        Image Loader
        :param img_path: path to the image file
        :return: image: np.ndarray
        """
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

    def save_image(
            self,
            file: np.ndarray,
            file_name: str,
            directory: os.path
    ):
        file_path = os.path.join(directory, file_name)
        cv2.imwrite(str(file_path), file)
        self.logger.info(f"Image saved as {file_path}.")

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
        # Enhance the image
        alpha = float(contrast)  # Contrast control (1.0-3.0)
        beta = float(brightness)  # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(crop_img, alpha=alpha, beta=beta)
        return adjusted

    def label_photo(
            self,
            image,
            contrast,
            brightness,
    ):
        # Crop and enhance the image before detection
        crop = self.crop_image(image)

        # Convert cropped image to grayscale for text detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = self.enhance_image(gray, contrast, brightness)

        # Denoise using Gaussian blur
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        binary, num_local = self.text_detection_tesseract(denoised)

        if len(num_local) > 1:
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else:
            marked_crop = crop
        return marked_crop, binary, num_local

    def simple_filter(self, num_loc):
        """Filter detections using only the whitelist"""
        return {k: v for k, v in num_loc.items() if k in self.number_whitelist}

    def text_detection_tesseract(self, file: np.ndarray):
        # Detect text regions using pytesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        detection_data = pytesseract.image_to_data(
            file,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Initialize a dictionary to store the locations of the numbers
        number_locations = {}
        for i in range(len(detection_data['text'])):
            num = detection_data['text'][i].strip()

            # Skip empty or non-numeric text
            if not num.replace('.', '', 1).isdigit():
                continue

            # Normalize decimal formatting
            if '.' in num:
                parts = num.split('.')
                if len(parts) != 2:  return None  # Handle multiple decimals
                if parts[1] == '0':  # Handle whole numbers
                    normalized = parts[0]
                else:
                    decimal_part = parts[1].rstrip('0')  # Handle decimal numbers
                    normalized = f"{parts[0]}.{decimal_part}" if decimal_part else parts[0]
                if normalized.startswith('.'):  # Add leading zero if needed
                    normalized = '0' + normalized
            else:
                normalized = num

            # Store only if in whitelist
            if normalized in self.number_whitelist:
                number_locations[normalized] = {
                    'coordinates': (
                        detection_data['left'][i],
                        detection_data['top'][i],
                        detection_data['width'][i],
                        detection_data['height'][i]
                    )
                }
        return file, number_locations

    def capture_colors(
            self,
            file: np.ndarray,
            num_locations
    ):
        """
        Read the color of a square shape underneath the numbers
        """
        (w, h) = (0, 0)
        n = len(num_locations)                              # Get the average size of the numbers

        if n == 0:
            self.logger.info('Nothing found!')
            return file, {}
        for num in num_locations:
            w += num_locations[num]['coordinates'][2]
            h += num_locations[num]['coordinates'][3]

        average_w, average_h = w//n, h//n
        for num in num_locations:
            roi_x1 = num_locations[num]['coordinates'][0]
            roi_y1 = num_locations[num]['coordinates'][1] + 2 * average_h
            roi_x2 = roi_x1 + average_w
            roi_y2 = roi_y1 + 2 * average_h

            # Extract ROI and calculate the average color of the ROI
            roi = file[roi_y1:roi_y2, roi_x1:roi_x2]
            avg_color_per_row = np.average(roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            num_locations[num]['color'] = avg_color

            # Highlight the number and region of interest (ROI) with a rectangle
            cv2.rectangle(file, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        return file, num_locations

    def aggregate_viz_positions(self, crop_image):
        """
        Process position records with advanced filtering for digits and decimals
        Featuring a two-phase aggregation with robust filtering
        """
        # Phase 1: Identify valid anchor points
        anchor_positions = self._identify_anchor_positions()

        # Phase 2: Filter and aggregate with clean median
        aggregated = {}
        for num in self.position_records.copy():
            valid_coords = self._filter_coordinates(
                np.array(self.position_records[num]),
                num,
                anchor_positions
            )
            count = len(valid_coords)
            if count == 0:
                del self.position_records[num]
                continue

            # Calculate robust median using IQR
            median_coord = self._robust_median(valid_coords)
            x, y, w, h = median_coord

            # Get color from original crop using stabilized position
            roi_y1 = median_coord[1] + 2 * median_coord[3]
            roi_y2 = median_coord[1] + 4 * median_coord[3]
            roi_x1 = median_coord[0]
            roi_x2 = median_coord[0] + median_coord[2]

            # Ensure ROI stays within image bounds
            roi = crop_image[
                  max(0, roi_y1):min(crop_image.shape[0], roi_y2),
                  max(0, roi_x1):min(crop_image.shape[1], roi_x2)
                  ]

            avg_color = (0, 0, 0)
            if roi.size > 0:
                avg_color_per_row = np.average(roi, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)

            aggregated[num] = {
                'coordinates': tuple(median_coord),
                'color': avg_color,
                'count': count
            }

        # Create visualization with counts
        marked_crop = crop_image.copy()
        for num_str, data in aggregated.items():
            x, y, w, h = data['coordinates']

            # Draw bounding box
            cv2.rectangle(marked_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create label text with number and count
            label_text = f"{num_str} ({data['count']})"

            # Calculate text size for proper positioning
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                marked_crop, (x, y - text_height - 10), (x + text_width, y - 10), (0, 0, 0), -1
            )

            # Add text with count
            cv2.putText(
                marked_crop, label_text,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
        return aggregated, marked_crop

    def _identify_anchor_positions(self):
        """Find reliable positions of multi-part numbers"""
        anchors = defaultdict(list)
        for num in self.position_records:
            if len(num) > 1 or '.' in num:
                coords = np.array(self.position_records[num])
                if len(coords) > 0:
                    anchors[num] = self._robust_median(coords)
        return anchors

    def _filter_coordinates(self, coords, current_num, anchors):
        """Remove coordinates conflicting with verified anchors"""
        if len(coords) == 0:
            return []

        # Check against all relevant anchors
        invalid_mask = np.zeros(len(coords), dtype=bool)

        # Get potential parent numbers from whitelist
        parent_numbers = self._get_parent_numbers(current_num)

        for parent_num in parent_numbers:
            if parent_num in anchors:
                anchor_pos = anchors[parent_num]
                # Vectorized distance calculation
                distances = np.sqrt(
                    (coords[:, 0] - anchor_pos[0]) ** 2 +
                    (coords[:, 1] - anchor_pos[1]) ** 2
                )
                invalid_mask |= distances < self._dynamic_threshold(parent_num)

        return coords[~invalid_mask]

    def _robust_median(self, coords):
        """Outlier-resistant median calculation using IQR"""
        if len(coords) < 3:
            return np.median(coords, axis=0).astype(int)

        q1 = np.percentile(coords, 25, axis=0)
        q3 = np.percentile(coords, 75, axis=0)
        iqr = q3 - q1

        inlier_mask = (
                (coords >= q1 - 1.5 * iqr) &
                (coords <= q3 + 1.5 * iqr)
        ).all(axis=1)

        return np.median(coords[inlier_mask], axis=0).astype(int)

    def _get_parent_numbers(self, num):
        """Get possible parent numbers from whitelist"""
        return [n for n in self.number_whitelist
                if num in n.replace('.', '') and n != num]

    def _dynamic_threshold(self, parent_num):
        """Adaptive threshold based on number format"""
        if '.' in parent_num:
            return self.max_x_variance * 1.5
        return self.max_x_variance * len(parent_num)

    def diff_image(
            self,
            img_path_dry: str,
            img_path_wet: str
    ) -> tuple:
        """
        TODO: complete documentation
        Returns:
            tuple: (heatmap_path, (center_x, center_y, diameter))
                   - heatmap_path: path to generated heatmap image
                   - coordinates: tuple of (x, y, diameter) or None
        """
        # Load and validate images
        img_dry = self.crop_image(cv2.imread(img_path_dry))
        img_wet = self.crop_image(cv2.imread(img_path_wet))

        wet_path = Path(img_path_wet)

        if img_dry is None or img_wet is None:
            self.logger.error("Failed to load one or both images")
            return (None, None)

        if img_dry.shape != img_wet.shape:
            self.logger.error("Image dimensions mismatch")
            return (None, None)

        # Calculate perceptual color difference
        lab_dry = cv2.cvtColor(img_dry, cv2.COLOR_BGR2LAB)
        lab_wet = cv2.cvtColor(img_wet, cv2.COLOR_BGR2LAB)

        delta_e = np.sqrt(
            np.square(lab_dry[:, :, 1].astype(np.float32) - lab_wet[:, :, 1]) +
            np.square(lab_dry[:, :, 2].astype(np.float32) - lab_wet[:, :, 2])
        )

        # Generate heatmap
        norm_diff = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        heatmap_path = wet_path.parent / f"{wet_path.stem}_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap)

        # Analyze change region
        blurred = cv2.GaussianBlur(delta_e, (51, 51), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

        if max_val < 5:
            self.logger.info("No significant changes detected")
            return (heatmap_path, None)

        # Calculate weighted centroid
        y_grid, x_grid = np.indices(delta_e.shape)
        weights = np.maximum(blurred - max_val * 0.5, 0)
        total_weight = np.sum(weights)

        if total_weight < 1e-6:
            return (heatmap_path, None)

        center_x = np.sum(x_grid * weights) / total_weight
        center_y = np.sum(y_grid * weights) / total_weight

        # Calculate effective diameter
        x_dev = np.sqrt(np.sum(weights * (x_grid - center_x) ** 2) / total_weight)
        y_dev = np.sqrt(np.sum(weights * (y_grid - center_y) ** 2) / total_weight)
        diameter = 2.3548 * np.sqrt(x_dev ** 2 + y_dev ** 2)  # FWHM conversion

        return (
            heatmap_path,
            (int(center_x), int(center_y), int(diameter))
        )

    def analysis_with_opt(
            self,
            image_path,  # Path to image
            step=25  # Default opt parameter
    ):
        """
        Returns:
            aggregated_numbers:
            crop:
            marked_crop:
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = analyzer.load_image(image_path)
        directory, file_name = os.path.split(image_path)
        file_name_no_extension = file_name[:-4]

        brightness_steps = range(-200, 201, step)  # Initialize results matrix of OCR
        contrast_steps = [i / 100 for i in range(0, 301, step)]  # 1.00-3.00 in steps
        opt_results = np.zeros((len(brightness_steps), len(contrast_steps)))
        opt_contrast, opt_brightness, max_num = 0, 0, 0

        crop = self.crop_image(image)               # Cropped color image
        detection_matrix = np.zeros(crop.shape[:2], dtype=np.uint16)

        for j_idx, j in enumerate(brightness_steps):  # Optimization loop
            for i_idx, i in enumerate(contrast_steps):
                contrast = i  # Get current contrast value
                _, _, num_loc = self.label_photo(image, contrast=contrast, brightness=j)
                current_count = len(num_loc)
                self.logger.info(f"Trying contrast={i}, brightness={j} found {current_count} numbers.")

                if num_loc:
                    for num in num_loc:
                        x, y, w, h = num_loc[num]['coordinates']
                        opt_results[j_idx, i_idx] = current_count  # Store results in OCR optimization matrix
                        detection_matrix[y:y + h, x:x + w] += 1  # Store results in heatmap
                        self.position_records[num].append((x, y, w, h))  # Add found positions to aggregated records
                if current_count >= max_num:  # Track optimal parameters (optional)
                    max_num = current_count

        if np.max(detection_matrix) > 0:  # Save the heatmap canvas
            # Normalize to 0-255 with higher counts = darker
            normalized = 255 - (detection_matrix / np.max(detection_matrix) * 255).astype(np.uint8)
        else:
            normalized = np.full_like(detection_matrix, 255, dtype=np.uint8)
        detection_viz = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel "white" background
        analyzer.save_image(detection_viz, f'{file_name_no_extension}_detections_{step=}.png', directory)

        # Generate heatmap of numbers OCR picks up vs contrast and brightness
        plt.figure(figsize=(8, 8))  # [contrast_min, contrast_max, brightness_min, brightness_max]
        plt.imshow(opt_results, aspect='auto', extent=(0.0, 3.0, -200, 200), origin='lower')
        plt.colorbar(label='Numbers Recognized')
        plt.xlabel('Contrast')
        plt.ylabel('Brightness')
        plt.title('Number Recognition Performance Heatmap')

        # Plot the data
        plt.xticks(np.arange(0.0, 3.0, 0.1))  # Add grid lines
        plt.yticks(np.arange(-200, 200, 10))
        plt.savefig(f'photos/{file_name_no_extension}_heatmap-{step=}.png', dpi=300, bbox_inches='tight')

        # Aggregate positions after all iterations
        aggregated_numbers, marked_crop = self.aggregate_viz_positions(crop)
        self.save_image(marked_crop, f'{file_name_no_extension}_aggregated_{step=}.png', directory)
        return aggregated_numbers, marked_crop

    def read_ph(
            self,
            read_coord: Tuple,
            read_width: int,
            image_path,  # Path to image
            aggregated_nums: Dict
    ):
        """
        This function will highlight the Region of Interest (ROI) to read the pH of the wet pH paper
        :param read_coord: (X,Y) coordinate of the top left corner of the ROI
        :param read_width: width the square-shaped ROI
        :param image_path: path to the target image
        :param aggregated_nums: The infor of known pH value and colors from the same image
        :returns:
            file: image with ROI highlighted
            ph: pH value
        """
        image = analyzer.load_image(image_path)
        crop = self.crop_image(image)
        directory, file_name = os.path.split(image_path)
        file_name_no_extension = file_name[:-4]

        # Use aggregated numbers for pH reading
        if len(aggregated_nums) >= 10:  # Reduced threshold for partial results
            try:
                # Highlight ROI and read the average color from it
                x1 = read_coord[0]
                x2 = read_coord[0] + read_width
                y1 = read_coord[1]
                y2 = read_coord[1] + read_width
                read_roi = crop[y1:y2, x1:x2]
                avg_color_per_row = np.average(read_roi, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                self.logger.info(f'The color of the ROI:'
                                 f'({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})')

                # Calculate the Euclidean distance between the ROI and all colors in aggregated data
                color_dist = {}
                for num in aggregated_nums:
                    num_color = aggregated_nums[num]["color"]
                    color_dist[num] = np.linalg.norm(avg_color - num_color)

                sorted_color_dist = dict(sorted(color_dist.items(), key=lambda item: item[1]))
                closest_ph = list(sorted_color_dist.keys())[0]
                self.logger.info(f"***\nThe closest pH value is {closest_ph}.\n***")

                crop_ph = crop.copy()
                cv2.rectangle(crop_ph, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(crop_ph, f"pH: {closest_ph}",
                            (read_coord[0], read_coord[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Save and return results
                self.save_image(crop_ph,f'{file_name_no_extension}_ph_result.png', directory)
                return closest_ph, aggregated_nums, crop_ph

            except Exception as e:
                self.logger.error(f"pH reading failed: {str(e)}")
                return str(e), aggregated_nums, crop
        return 'Insufficient detections', aggregated_nums, crop


if __name__ == "__main__":
    analyzer = PhotoAnalyzer()

    # Macbook setting override
    # pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

    path_to_image_dry = ("photos/capture_20250225-160453_200200200.jpg")
    path_to_image_wet = ("photos/capture_20250225-160539_200200200.jpg")

    _, ROI = analyzer.diff_image(path_to_image_dry, path_to_image_wet)
    print(f"ROI_X = {ROI[0]}, ROI_Y = {ROI[1]}, diameter = {ROI[2]}")

    for opt_step in [5]:
        try:
            aggregate_nums, labelled_crop = analyzer.analysis_with_opt(path_to_image_wet, step=opt_step)
            if aggregate_nums:
                closest_ph, aggregated_nums, crop_ph = analyzer.read_ph(
                (ROI[0]-ROI[2]//4, ROI[1]-ROI[2]//4), ROI[2]//2,
                #(75,75),40,
                path_to_image_wet, aggregate_nums
                )
                analyzer.logger.info(f"The pH = {closest_ph}.")
            else:
                analyzer.logger.error("No valid numbers detected")
        except Exception as e:
            analyzer.logger.error(f"Analysis failed: {str(e)}")
