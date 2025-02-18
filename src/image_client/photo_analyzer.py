import os
import cv2
import numpy as np
import pytesseract
from typing import Dict, Tuple
from sdl_utils import get_logger

# Processing parameters
# Cropping parameters. Image will be cropped first
startY, endY, startX, endX = 800, 2000, 1000, 4200
width = (endX - startX) // 4
height = (endY - startY) // 4

# For Windows computers, do the following
# Please install tesseract first  https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PhotoAnalyzer:
    def __init__(
            self,
            logger = None
    ):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        # Create the logger and file handler
        logger = get_logger("ImageAnalyser")
        return logger

    def load_image(self, image_path: os.path):
        """
        Image Loader
        :param image_path: path to the image file
        :return: image: np.ndarray
        """
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error("Error: Image not found or unable to load.")
        else:
            self.logger.info(f"Image loaded from {image_path}.\nImage dimensions: {image.shape}")
        return image

    def view_image(self, file: np.ndarray):
        cv2.imshow('Image', file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(
            self,
            file: np.ndarray,
            file_name: str,
            directory: os.path
    ):
        file_path = os.path.join(directory, file_name)
        cv2.imwrite(str(file_path), file)
        print(f"Image saved as {file_path}.")

    def crop_image(
            self,
            file: np.ndarray
    ):
        # Define the cropping coordinates
        crop_box = file[startY:endY, startX:endX]
        crop_img = cv2.resize(crop_box, (width, height), interpolation=cv2.INTER_AREA)
        return crop_img

    def enhance_image(
            self,
            crop_img: np.ndarray,
            contrast: float,
            brightness: int
    ):
        # Enhance the image
        alpha = float(contrast)  # Contrast control (1.0-3.0)
        beta = float(brightness)  # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(crop_img, alpha=alpha, beta=beta)
        return adjusted

    def text_detection(
            self,
            file: np.ndarray
    ):
        # Convert cropped image to grayscale for text detection
        grey = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        # Detect text regions using pytesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        detection_data = pytesseract.image_to_data(
            grey,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Initialize a dictionary to store the locations of the numbers
        number_locations = {}
        for i in range(len(detection_data['text'])):
            num = detection_data['text'][i]
            x = detection_data['left'][i]
            y = detection_data['top'][i]
            w = detection_data['width'][i]
            h = detection_data['height'][i]

            if num.isdigit() and (1 <= int(num) <= 14):
                # print(f"Number {num} found in the image, at ", f"position:({x},{y})")
                number_locations[num] = {
                    'coordinates': (int(x), int(y), int(w), int(h))
                }
        return grey, number_locations

    def capture_colors(
            self,
            file: np.ndarray,
            num_locations
    ):

        (W, H) = (0, 0)

        # Get the average size of the numbers
        n = len(num_locations)
        if n == 0:
            self.logger.info('Nothing found!')
            return file, {}

        for num in num_locations:
            W += num_locations[num]['coordinates'][2]
            H += num_locations[num]['coordinates'][3]

        average_w = W//n
        average_h = H//n
        self.logger.info(f"OCR parameters {average_w=}, {average_h=}")

        for num in num_locations:
            roi_x1 = num_locations[num]['coordinates'][0]
            roi_y1 = num_locations[num]['coordinates'][1] + 2 * average_h
            roi_x2 = num_locations[num]['coordinates'][0] + average_w
            roi_y2 = num_locations[num]['coordinates'][1] + 4 * average_h

            # Extract ROI and calculate the average color of the ROI
            roi = file[roi_y1:roi_y2, roi_x1:roi_x2]
            avg_color_per_row = np.average(roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            num_locations[num]['color'] = avg_color

            # Highlight the number and region of interest (ROI) with a rectangle
            cv2.rectangle(file, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        return file, num_locations

    def read_ph(
            self,
            file: np.ndarray,
            coordinate: Tuple,
            width: int,
            num_locations: Dict
    ):
        """
        This function will highlight the Region of Interest (ROI) to read the pH of the wet pH paper
        :param file: CROPPED COLOR image file to be read from
        :param coordinate: (X,Y) coordinate of the top left corner of the ROI
        :param width: width the square-shaped ROI
        :param num_locations: The infor of known pH value and colors from the same image
        :returns:
            file: image with ROI highlighted
            ph: pH value
        """
        # Highlight ROI and read the average color from it
        x1 = coordinate[0]
        x2 = coordinate[0] + width
        y1 = coordinate[1]
        y2 = coordinate[1] + width

        read_roi = file[y1:y2, x1:x2]
        avg_color_per_row = np.average(read_roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        self.logger.info(f'The color of the ROI:'
                         f'({int(avg_color[0])}, {int(avg_color[1])}, {int(avg_color[2])})')

        # Calculate the Euclidean distance between the ROI and all colors with known pH
        color_dist = {}
        sorted_color_dist = {}
        if len(num_locations) == 1:
            return file, None

        for num in num_locations:
            num_color = num_locations[num]['color']
            color_dist[num] = np.linalg.norm(avg_color - num_color)

        sorted_color_dist = dict(sorted(color_dist.items(), key=lambda item: item[1]))
        closest_ph = list(sorted_color_dist.keys())[0]
        self.logger.info(f"***\nThe closest pH value is {closest_ph}.\n***")

        cv2.rectangle(file, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return file, closest_ph


    def label_photo(
            self,
            image,
            contrast,
            brightness
    ):
        # Crop and enhance the image before detection
        crop = self.crop_image(image)
        enhance = self.enhance_image(crop, contrast, brightness)
        grey, num_local = self.text_detection(enhance)

        if len(num_local) !=0:
            marked_crop, num_local = self.capture_colors(crop, num_local)
        else:
            marked_crop = crop
        return marked_crop, num_local


    def diff_image(
            self,
            img_path_dry: str,
            img_path_wet: str
    ):
        directory, file_name = os.path.split(img_path_wet)
        img_dry = cv2.imread(img_path_dry)
        img_wet = cv2.imread(img_path_wet)
        if img_dry.shape == img_wet.shape:
            diff = cv2.absdiff(img_dry, img_wet)
            grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            enhance_diff = self.enhance_image(grey_diff, 2.0, 0)

            # Save the result
            self.save_image(enhance_diff, 'diff_' + file_name, directory)
        else:
            self.logger.info("Images are not the same size.")


if __name__ == "__main__":
    analyser = PhotoAnalyzer()

