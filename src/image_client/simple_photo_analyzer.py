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
        with open(script_dir / 'image_client_settings.yaml', 'r') as file:
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

    def find_closest_ph(self, calibration_sample_paths, unknown_sample_path):
        """
        Compare unknown sample to known pH samples
        Args:
            calibration_sample_paths: Dict of {"pH": "image_path"}
            unknown_sample_path: Path to test image
        Returns:
            (closest_ph, color_distances)
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
                self.logger.info(f"Unknown color is {unknown_color}")
            except Exception as e:
                raise ValueError(f"Unknown sample analysis failed: {str(e)}")

            # Calculate distances with validation
            distances = {}
            unknown_array = np.array(unknown_color)
            for ph, ref_color in reference_colors.items():
                self.logger.debug(f"The reference color for pH={ph} is {ref_color}.")
                ref_array = np.array(ref_color)
                distances[ph] = np.linalg.norm(ref_array - unknown_array)

            closest_ph = min(distances, key=distances.get)
            output_path = self.save_annotated_image(unknown_data, closest_ph)
            return closest_ph, distances, output_path

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

    try:
        closest_ph, distances, output_path = analyzer.find_closest_ph(
            known_samples, unknown_sample
        )
        analyzer.logger.info("All distances:")
        for ph, distance in distances.items():
            analyzer.logger.info(f"pH {ph}: {distance:.2f}")
        analyzer.logger.info(f"Closest pH: {closest_ph}")
        analyzer.logger.info(f"Annotated image saved to: {output_path}")
    except Exception as e:
        analyzer.logger.error(f"Analysis failed: {str(e)}")