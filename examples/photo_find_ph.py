import os
from photo_client import ImageAnalyzer


if __name__ == "__main__":

    analyzer = ImageAnalyser()
    image_path = ('photos/2025-02-11_20-53-30.jpg')

    image = analyzer.load_image(image_path)
    directory, file_name = os.path.split(image_path)

    opt_contrast = 0
    opt_brightness = 0
    max_num = 0
    for j in range(0, 100, 5):
        for i in range(100, 301, 5):

            _, num_loc = analyzer.label_photo(image, contrast=i / 100, brightness=j)
            print(f"Trying contrast={i / 100}, brightness={j}, found {len(num_loc)} numbers")

            if len(num_loc) >= max_num:
                max_num = len(num_loc)
                opt_contrast = i / 100
                opt_brightness = j
            else:
                continue

    print(
        f"***\nThe optimal contrast for OCR is {opt_contrast}\n"
        f"The optimal brightness for OCR is {opt_brightness}\n"
        f"Up to {max_num} numbers recognized!\n***"
    )

    marked_crop, num_loc = analyzer.label_photo(image, contrast=opt_contrast, brightness=opt_brightness)
    analyzer.read_ph(marked_crop, (110, 110), 30, num_loc)
    analzer.save_image(marked_crop, 'crop_' + file_name, directory)
