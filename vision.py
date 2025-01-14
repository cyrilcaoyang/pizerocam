import os
import cv2
import numpy as np
import pytesseract
import json

# Setting the path to the trainning data
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'


# Function to extract text from image using pytesseract
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


# Function to calculate the average color of a region in the image
def average_color(image, x, y, w, h):
    region = image[y:y + h, x:x + w]

    if region.size == 0: return [0, 0, 0]

    avg_color_per_row = np.average(region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color.tolist()


# Load the image
image_path = 'image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert image frp, BGR to RGB for Tesseract
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
custom_config = r'-c tessedit_char_whitelist=0123456789. --psm 6 --oem 0'

# Convert image to grayscale for text detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect text regions using pytesseract
detection_data = pytesseract.image_to_data(
    rgb,
    config=custom_config,
    output_type=pytesseract.Output.DICT
)

# Initialize list to store color block data
color_blocks_data = []

# Iterate through detected text regions
for i in range(len(detection_data['text'])):

    if int(detection_data['conf'][i]) > 70:  # Confidence threshold
        x, y, w, h = detection_data['left'][i], detection_data['top'][i], detection_data['width'][i], \
        detection_data['height'][i]
        text = detection_data['text'][i].strip()

        # Calculate average color of the region below the text (assuming color block is below the text)
        avg_color = average_color(image, x, y + h + 5, w, h)  # Adjust y + h + 5 as needed

        # Store the data in the list
        color_blocks_data.append({
            'text': text,
            'average_color': avg_color
        })

# Save the data to a JSON file
output_path = 'color_blocks_data.json'
with open(output_path, 'w') as json_file:
    json.dump(color_blocks_data, json_file, indent=4)

print(f"Color blocks data saved to {output_path}.")

