"""
A simple photo analyzer that give pH readings with hard-coded positions.

Author: Yang Cao, Acceleration Consortium
Email: yangcyril.cao@utoronto.ca
Version: 0.1
"""

import cv2
import yaml
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
ROI_X: 120
ROI_Y: 120
ROI_W:  20
ROI_H:  20

class SimplePhotoAnalyzer:
    def __init__(self):

        # self.logger = logger
        script_dir = Path(__file__).resolve().parent  # Get the directory where this script is located
        with open(script_dir / 'image_req_client_settings.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # Load whitelist and tolerance of OCR
        self.roi_x = data.get('ROI_X')
        self.roi_y = data.get('ROI_Y')
        self.roi_w = data.get('ROI_W', 20)
        self.roi_h = data.get('ROI_H', 20)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("Simple pH Analyzer")
        self.reference_output_dir = "reference_annotations"  # New directory for reference images

    def get_roi_data(self, image_path):
        """Extract average color from fixed ROI"""
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Could not load image: {image_path}")
            raise ValueError(f"Could not load image: {image_path}")
            # Get image dimensions

        h, w = img.shape[:2]

        # Calculate valid ROI coordinates
        x1 = max(0, min(self.roi_x, w - 1))
        y1 = max(0, min(self.roi_y, h - 1))
        x2 = min(x1 + self.roi_w, w)
        y2 = min(y1 + self.roi_h, h)

        # Verify valid ROI dimensions
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            raise ValueError(
                f"Invalid ROI for {image_path}\n"
                f"Image size: {w}x{h}\n"
                f"Adjusted ROI: ({x1},{y1}) to ({x2},{y2})"
            )
        roi = img[y1:y2, x1:x2]

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(roi_rgb, axis=(0, 1)).astype(np.uint8)          # [R G B]
        formatted_color = avg_color.tolist()                                # [R, G, B]
        self.logger.info(f"The average color of the ROI is {formatted_color} in {image_path}.")
        return {
            'color': formatted_color,
            'coordinates': (x1, y1, x2, y2),
            'image': img,  # Return the loaded image for annotation
            'path': image_path  # Pass on the image path
        }

    def save_annotated_image(
            self, image_data, ph_value,
            is_reference=False,
            output_path=None
    ):
        """Save annotated image with ROI and pH label"""
        img = image_data['image'].copy()
        x1, y1, x2, y2 = image_data['coordinates']

        # Changed label based on reference status
        label = f"Ref pH={ph_value}" if is_reference else f"Result pH: {ph_value}"
        color = (0, 0, 0) if not is_reference else (0, 0, 0)    # Red for references
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 10)
        font_scale = 4
        thickness = 10

        # Calculate text position
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)

        # Ensure text stays within image bounds
        text_y = max(y1 - 10, text_h + 10)
        text_x = max(x1, 10)

        cv2.putText(img, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness)

        # Generate output path in same folder
        if output_path is None:
            original_path = Path(image_data['path'])
            prefix = "ref_" if is_reference else "analyzed_"
            output_path = original_path.with_name(f"{prefix}{original_path.name}")

        cv2.imwrite(str(output_path), img)
        self.logger.info(f"Saved annotated image to: {output_path}")
        return output_path

    def convert_color_space(self, rgb_color, color_space='rgb'):
        """
        Convert RGB color to specified color space
        Args:
            rgb_color: RGB color as [R, G, B] list/array
            color_space: 'rgb', 'lab', or 'hsv'
        Returns:
            Converted color as numpy array
        """
        rgb_array = np.array(rgb_color, dtype=np.uint8).reshape(1, 1, 3)
        
        if color_space.lower() == 'rgb':
            return rgb_array.flatten().astype(np.float32)
        elif color_space.lower() == 'lab':
            # Convert RGB to LAB
            lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
            return lab.flatten().astype(np.float32)
        elif color_space.lower() == 'hsv':
            # Convert RGB to HSV
            hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            return hsv.flatten().astype(np.float32)
        else:
            raise ValueError(f"Unsupported color space: {color_space}. Use 'rgb', 'lab', or 'hsv'.")

    def interpolate_ph(self, distances):
        """
        Interpolate pH value to one decimal place based on two closest references
        Args:
            distances: Dict of {pH: distance}
        Returns:
            Interpolated pH value rounded to 1 decimal place
        """
        # Sort pH values by distance
        sorted_phs = sorted(distances.items(), key=lambda x: x[1])
        
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
        # The closer the distance, the more weight it gets
        weight1 = 1.0 / dist1
        weight2 = 1.0 / dist2
        total_weight = weight1 + weight2
        
        interpolated_ph = (ph1 * weight1 + ph2 * weight2) / total_weight
        return round(interpolated_ph, 1)

    def find_closest_ph(self, calibration_sample_paths, unknown_sample_path, 
                       color_space='rgb', interpolate=True):
        """
        Compare unknown sample to known pH samples using specified color space
        Args:
            calibration_sample_paths: Dict of {"pH": "image_path"}
            unknown_sample_path: Path to test image
            color_space: 'rgb', 'lab', or 'hsv' - color space for distance calculation
            interpolate: If True, interpolate pH to 1 decimal place; if False, return closest match
        Returns:
            (estimated_ph, color_distances, output_path)
        """
        try:
            # Load reference colors
            reference_colors = {}
            for ph, path in calibration_sample_paths.items():
                data = self.get_roi_data(Path(path))
                reference_colors[ph] = data['color']

                # Save annotated reference image
                self.save_annotated_image(
                    image_data=data,
                    ph_value=ph,
                    is_reference=True
                )

            # Process unknown sample
            try:
                unknown_data = self.get_roi_data(Path(unknown_sample_path))
                unknown_color = unknown_data['color']
                self.logger.info(f"Unknown RGB color: {unknown_color}")
            except Exception as e:
                raise ValueError(f"Unknown sample analysis failed: {str(e)}")

            # Convert to specified color space
            unknown_converted = self.convert_color_space(unknown_color, color_space)
            self.logger.info(f"Color space: {color_space.upper()}, Converted unknown: {unknown_converted}")

            # Calculate distances in the specified color space
            distances = {}
            for ph, ref_color in reference_colors.items():
                ref_converted = self.convert_color_space(ref_color, color_space)
                self.logger.debug(f"pH={ph} reference in {color_space.upper()}: {ref_converted}")
                distances[ph] = np.linalg.norm(ref_converted - unknown_converted)

            # Get pH estimate
            if interpolate:
                estimated_ph = self.interpolate_ph(distances)
                self.logger.info(f"Interpolated pH estimate: {estimated_ph}")
            else:
                estimated_ph = min(distances, key=distances.get)
                self.logger.info(f"Closest pH match: {estimated_ph}")
            
            output_path = self.save_annotated_image(unknown_data, estimated_ph)
            return estimated_ph, distances, output_path

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise


if __name__ == "__main__":
    analyzer = SimplePhotoAnalyzer()

    known_samples = {
        1.0: "photos/simple_workflow/capture_20250226_200200200_1.jpg",
        4.0: "photos/simple_workflow/capture_20250226_200200200_4.jpg",
        7.0: "photos/simple_workflow/capture_20250226_200200200_7.jpg",
        10.0: "photos/simple_workflow/capture_20250226_200200200_10.jpg",
        12.5: "photos/simple_workflow/capture_20250226_200200200_12.jpg",
    }

    unknown_sample = "photos/simple_workflow/capture_20250226_200200200_unknown.jpg"

    # Example: Try different color spaces
    for color_space in ['rgb', 'lab', 'hsv']:
        try:
            analyzer.logger.info(f"\n{'='*60}")
            analyzer.logger.info(f"Testing with {color_space.upper()} color space")
            analyzer.logger.info(f"{'='*60}")
            
            estimated_ph, distances, output_path = analyzer.find_closest_ph(
                known_samples, unknown_sample,
                color_space=color_space,  # Choose: 'rgb', 'lab', or 'hsv'
                interpolate=True  # Set to False for exact match only
            )
            
            analyzer.logger.info("All distances:")
            for ph, distance in distances.items():
                analyzer.logger.info(f"  pH {ph}: {distance:.2f}")
            analyzer.logger.info(f"Estimated pH: {estimated_ph}")
            analyzer.logger.info(f"Annotated image saved to: {output_path}")
        except Exception as e:
            analyzer.logger.error(f"Analysis failed: {str(e)}")