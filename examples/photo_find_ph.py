import os
from image_client import PhotoAnalyzer


if __name__ == "__main__":

    analyzer = PhotoAnalyzer()
    image_path = ('photos/2025-02-11_20-51-04.jpg')
    # image_path = ('photos/interval_64/capture_20250217-164227_127255127.jpg')

    image = analyzer.load_image(image_path)

    directory, file_name = os.path.split(image_path)

    opt_contrast = 0
    opt_brightness = 0

    max_num = 0
    for j in range(0, 201, 50):
        for i in range(200, 301, 5):

            _, num_loc = analyzer.label_photo(image, contrast=i / 100, brightness=j)
            analyzer.logger.info(f"Trying contrast={i / 100}, brightness={j})"
                                 f" found {len(num_loc) if num_loc else 0} numbers")

            current_count = len(num_loc) if num_loc else 0
            if current_count >= max_num:
                max_num = len(num_loc)
                opt_contrast = i / 100
                opt_brightness = j
                analyzer.logger.info(f"New best! digits are {num_loc}")
            else:
                continue

    analyzer.logger.info(f"***\nThe optimal contrast for OCR is {opt_contrast}\n")
    analyzer.logger.info(f"The optimal brightness for OCR is {opt_brightness}\n")
    analyzer.logger.info(f"Up to {max_num} numbers recognized!\n***")

    marked_crop, num_loc = analyzer.label_photo(image, contrast=opt_contrast, brightness=opt_brightness)

    analyzer.logger.info(f"Found {len(num_loc) if num_loc else 0} numbers")

    analyzer.read_ph((110, 110), 30, marked_crop, num_loc)
    analyzer.save_image(marked_crop, 'crop_' + file_name, directory)
