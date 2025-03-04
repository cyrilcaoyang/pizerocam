"""
Photo analyzer that give pH readings.
1. Local method:
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
import cv2
import yaml
import re
import numpy as np
import pytesseract
from pprint import pprint
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from sdl_utils import get_logger
from google.cloud import vision

# Cropping parameters. Image will be cropped first TODO: add soft cropping with edge detection
startY, endY, startX, endX = 600, 2000, 1000, 4400
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

        self.path_tesseract = data['Path_Tesseract']                    # For tesseract OCR use case only
        pytesseract.pytesseract.tesseract_cmd = self.path_tesseract
        self.expected_rows = data.get('Expected_Rows', 2)
        self.max_y_variance = data.get('Max_Y_Var', 15)
        self.max_x_variance = data.get('Max_X_Var', 30)
        self.min_row_density = data.get('Min_Row_Density', 0.7)

        self.number_whitelist = set(map(str, data['PH_Whitelist']))     # Load whitelist and tolerance of OCR
        self.logger.info(f"Whitelist loaded as: {self.number_whitelist}")


    @staticmethod
    def _setup_logger():
        logger = get_logger("PhotoAnalyzerLogger")                      # Create the logger and file handler
        return logger

    def load_image(self, img_path):
        """
        Image Loader
        :param img_path: path to the image file
        :returns: image: np.ndarray
        """
        try:
            color_image = cv2.imread(img_path)
            self.logger.info(f"Image loaded from {img_path}.\nImage dimensions: {color_image.shape}")
            return color_image
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
            filename: str,
            dir: os.path
    ):
        file_path = os.path.join(dir, filename)
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
        # Enhance the grey image for Tesseract
        adjusted = cv2.convertScaleAbs(crop_img, alpha=float(contrast), beta=float(brightness))
        return adjusted

    def text_detection_tesseract(
            self,
            file: np.ndarray
    ):
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

            if not num.replace('.', '', 1).isdigit(): continue          # Skip empty or non-numeric text
            if '.' in num:                                              # Normalize decimal formatting
                parts = num.split('.')
                if len(parts) != 2:  return None                        # Handle multiple decimals
                if parts[1] == '0':                                     # Handle whole numbers
                    normalized = parts[0]
                else:
                    decimal_part = parts[1].rstrip('0')                 # Handle decimal numbers
                    normalized = f"{parts[0]}.{decimal_part}" if decimal_part else parts[0]
                if normalized.startswith('.'):                          # Add leading zero if needed
                    normalized = '0' + normalized
            else: normalized = num

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
        return number_locations

    def text_detection_google_vision(self, image: np.ndarray):
        # Setting up google cloud API
        client = vision.ImageAnnotatorClient()

        # Convert image to bytes
        _, encoded_image = cv2.imencode(".jpg", image)
        image_bytes = encoded_image.tobytes()

        # Create the Vision API Image object
        image_obj = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image_obj)

        ph_values = []
        ph_pattern = re.compile(r'^\d+([.,]\d)?$')  # Match "7", "7.5", "7,5"
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        if ph_pattern.match(word_text):
                            normalized = word_text.replace(',', '.')    # Normalize to "X.X" format
                            if '.' not in normalized:
                                normalized += ".0"
                            elif len(normalized.split('.')[1]) > 1:
                                normalized = normalized.split('.')[0] + '.' + normalized.split('.')[1][0]

                            if normalized in self.number_whitelist:
                                vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                                ph_values.append({
                                    'ph_value': normalized,
                                    'bounding_box': vertices
                                })
        if response.error.message:
            raise Exception(f'{response.error.message}')
        return ph_values

    def capture_colors(
            self,
            file: np.ndarray,
            num_locations
    ):
        """
        Read the color of a square shape underneath the numbers
        :para:
        :returns:
        """
        (w, h, n) = (0, 0, len(num_locations))
        n = len(num_locations)

        if n == 0:
            self.logger.info('Nothing found!')
            return file, {}
        for num in num_locations:
            w += num_locations[num]['coordinates'][2]
            h += num_locations[num]['coordinates'][3]
        average_w, average_h = w//n, h//n                           # Getting average size of the numbers

        for num in num_locations:
            roi_x1 = num_locations[num]['coordinates'][0]           # Defining the ROIs
            roi_y1 = num_locations[num]['coordinates'][1] + 2 * average_h
            roi_x2 = roi_x1 + average_w
            roi_y2 = roi_y1 + 2 * average_h

            roi = file[roi_y1:roi_y2, roi_x1:roi_x2]                # Calculate the average color of the ROI
            avg_color_per_row = np.average(roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            num_locations[num]['color'] = avg_color
            cv2.rectangle(file, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) # highlight ROI with numbers
        return file, num_locations

    def find_num_loc_colors_tess(
            self,
            image,
            contrast,
            brightness,
    ):
        """
        Get the locations and colors of numbers using tesseract
        """
        crop = self.crop_image(image)                                   # Crop the image
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)                   # Convert image to greyscale for OCR
        enhanced_grey = self.enhance_image(grey, contrast, brightness)
        denoised_grey = cv2.GaussianBlur(enhanced_grey, (5, 5), 0)   # Denoise using Gaussian blur
        num_local = self.text_detection_tesseract(denoised_grey)        # OCR w/ tesseract, num_local has no color info

        if len(num_local) > 1:                                          # Update num_local to include color info
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else: marked_crop = crop
        return marked_crop, denoised_grey, num_local

    def find_num_loc_colors_cloud(
            self,
            image,
            contrast,
            brightness,
    ):
        """
        Get the locations and colors of numbers using Google Cloud Vision API # TODO
        """
        crop = self.crop_image(image)                                   # Crop the image
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)                   # Convert image to greyscale for OCR
        enhanced_grey = self.enhance_image(grey, contrast, brightness)
        denoised_grey = cv2.GaussianBlur(enhanced_grey, (5, 5), 0)   # Denoise using Gaussian blur
        num_local = self.text_detection_tesseract(denoised_grey)        # OCR w/ tesseract, num_local has no color info

        if len(num_local) > 1:                                          # Update num_local to include color info
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else: marked_crop = crop
        return marked_crop, denoised_grey, num_local

    def aggregate_viz_positions(self, crop_image):
        """
        Process position records with advanced filtering for digits and decimals
        Featuring a two-phase aggregation with robust filtering
        """
        anchor_positions = self._identify_anchor_positions()            # Identify valid anchor points
        aggregated = {}                                                 # Filter and aggregate with clean median
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
            roi_y1 = y + 2 * h
            roi_y2 = y + 4 * h
            roi_x1 = x
            roi_x2 = x + w

            roi = crop_image[
                  max(0, roi_y1):min(crop_image.shape[0], roi_y2),      # Ensure ROI stays within image bounds
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
            cv2.rectangle(marked_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            label_text = f"{num_str} ({data['count']})"                         # Create label text

            (text_width, text_height), _ = cv2.getTextSize(      # Calculate text size for proper positioning
                label_text, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 2
            )

            # Draw background rectangle for text
            # cv2.rectangle(marked_crop, (x, y - text_height - 10), (x + text_width, y - 10), (0, 0, 0), -1)
            cv2.putText(marked_crop, label_text,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
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
        """
        Remove coordinates conflicting with verified anchors
        """
        if len(coords) == 0: return []

        invalid_mask = np.zeros(len(coords), dtype=bool)            # Check against all relevant anchor
        parent_numbers = self._get_parent_numbers(current_num)      # Get potential parent numbers from whitelist

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
        :returns:
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
        diameter = 2.3548 * np.sqrt(x_dev ** 2 + y_dev ** 2)                  # FWHM conversion
        return (
            heatmap_path,
            (int(center_x), int(center_y), int(diameter))
        )

    def tesseract_ocr_optimization_flow(
            self,
            color_image: np.ndarray,      # Loaded color image as numpy array
            step=25                 # Default opt parameter
    ):
        """
        returns:
            aggregated_numbers: The information of known pH value and colors from the same image
            marked_crop: np.ndarray, the cropped color image with pH numbers marked on them
            ocr_detection_viz: np.ndarray, visualization of the OCR mapped on cropped grey scale image
            opt_heatmap: Image object (plot can be save with matplotlib)
        """
        crop = self.crop_image(color_image)                                       # Cropped color image
        detection_matrix = np.zeros(crop.shape[:2], dtype=np.uint16)

        brightness_steps = range(-200, 201, step)                           # Initialize results matrix of OCR
        contrast_steps = [i / 100 for i in range(0, 301, step)]             # 1.00-3.00 in steps
        opt_results = np.zeros((len(brightness_steps), len(contrast_steps)))
        opt_contrast, opt_brightness, max_num = 0, 0, 0

        for j_idx, j in enumerate(brightness_steps):  # Optimization loop
            for i_idx, i in enumerate(contrast_steps):
                contrast = i  # Get current contrast value
                _, _, num_loc = self.find_num_loc_colors_tess(color_image, contrast=contrast, brightness=j)
                current_count = len(num_loc)
                self.logger.info(f"Trying contrast={i}, brightness={j} found {current_count} numbers.")

                if num_loc:
                    for num in num_loc:
                        x, y, w, h = num_loc[num]['coordinates']
                        opt_results[j_idx, i_idx] = current_count           # Store results in OCR optimization matrix
                        detection_matrix[y:y + h, x:x + w] += 1             # Store results in heatmap
                        self.position_records[num].append((x, y, w, h))     # Add found positions to aggregated records
                if current_count >= max_num:                                # Track optimal parameters (optional)
                    max_num = current_count

        # Visualize the optimization of contrast and brightness using a heatmap
        opt_heatmap, axis = plt.subplots(figsize=(8, 8))
        im = axis.imshow(opt_results, aspect='auto', extent=(0.0, 3.0, -200, 200), origin='lower')
        plt.colorbar(im, label='Numbers Recognized')
        axis.set_xlabel('Contrast')
        axis.set_ylabel('Brightness')
        axis.set_title('Number Recognition Performance Heatmap')
        axis.set_xticks(np.arange(0.0, 3.0, 0.1))                                # Add grid lines to plot
        axis.set_yticks(np.arange(-200, 200, 10))

        # Map OCR results onto cropped images, normalize to 0-255 with higher counts = darker
        if np.max(detection_matrix) > 0:
            normalized = 255 - (detection_matrix / np.max(detection_matrix) * 255).astype(np.uint8)
        else:
            normalized = np.full_like(detection_matrix, 255, dtype=np.uint8)
        ocr_detection_viz = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)       # Convert to 3-channel "white" background

        aggregated_numbers, marked_crop = self.aggregate_viz_positions(crop) # Aggregate positions after all iterations
        return aggregated_numbers, marked_crop, ocr_detection_viz, opt_heatmap

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
        :param aggregated_nums: The information of known pH value and colors from the same image
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
                return closest_ph, crop_ph

            except Exception as e:
                self.logger.error(f"pH reading failed: {str(e)}")
                return str(e), crop
        return 'Insufficient detections', crop


if __name__ == "__main__":
    analyzer = PhotoAnalyzer()

    # Macbook setting override
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

    # Load images
    path_to_image_dry = ("photos/capture_20250228-143528_255255255.jpg")
    path_to_image_wet = ("photos/capture_20250228-143550_255255255.jpg")
    color_image = cv2.imread(path_to_image_wet)

    if not os.path.exists(path_to_image_wet):
        analyzer.logger.error(f"Image not found: {path_to_image_wet}")
        raise FileNotFoundError(f"Image not found: {path_to_image_wet}")
    directory, file_name = os.path.split(path_to_image_wet)
    file_name_no_extension = file_name[:-4]

    # Find ROI using contrast of wet and dry images
    _, ROI = analyzer.diff_image(path_to_image_dry, path_to_image_wet)
    analyzer.logger.info(f"ROI_X = {ROI[0]}, ROI_Y = {ROI[1]}, diameter = {ROI[2]}")

    ## USE CASE 1: USE TESSERACT
    # Optimize the OCR parameters
    steps = [40]
    for step in steps:
        try:
            aggregate_nums, labelled_crop, ocr_viz, opt_heat_map = (
                analyzer.tesseract_ocr_optimization_flow(color_image, step=step))
            analyzer.save_image(labelled_crop, f'{file_name_no_extension}_labelled_{step=}.png', directory)
            analyzer.save_image(ocr_viz, f'{file_name_no_extension}_detections_{step=}.png', directory)
            opt_heat_map.savefig(
                f'photos/{file_name_no_extension}_opt_heatmap-{step=}.png', dpi=300, bbox_inches='tight'
            )

            if aggregate_nums:
                pprint(aggregate_nums)
                # result format:
                # {'1.5': {'color': array([125.77467105, 33.32154605, 234.74095395]),
                #          'coordinates': (np.int64(1053), np.int64(104), np.int64(64), np.int64(38)),
                #          'count': 2},
                #  '2.5':...
                closest_ph, crop_ph = analyzer.read_ph(
                (150,150),80,
                path_to_image_wet, aggregate_nums
                )
                analyzer.logger.info(f"The pH = {closest_ph}.")
            else:
                analyzer.logger.error("No valid numbers detected")
        except Exception as e:
            analyzer.logger.error(f"Analysis failed: {str(e)}")

    # Google cloud API:
    image = analyzer.load_image(path_to_image_wet)
    google_pH = analyzer.text_detection_google_vision(image=image)
    # result in this format:
    # [{'bounding_box': [(2065, 820), (2255, 818), (2256, 899), (2066, 901)], 'ph_value': '0.0'},
    #  {'bounding_box': [(2402, 817), (2585, 815), (2586, 895), (2403, 897)], 'ph_value': '0.5'},
    #  ...
    pprint(google_pH)