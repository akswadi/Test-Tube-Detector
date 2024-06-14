# Image Processing Scripts

## Crop and Rotate Images

This script processes images by cropping and rotating them.

### What the Script Does

1. **Read Images**: It reads all images with `.jpg`, `.jpeg`, or `.png` extensions from the specified input directory (`input_folder`).
2. **Rotate Images**: Each image is rotated 90 degrees counterclockwise.
3. **Crop Images**: After rotation, the image is cropped using the specified coordinates (`crop_coords`).
4. **Save Processed Images**: The cropped and rotated images are saved to the specified output directory (`output_folder`).

### Usage Instructions

1. Place the images you want to process in the `4x_images` directory.
2. Adjust the `crop_coords` variable to specify the cropping area (in the format `(x1, y1, x2, y2)`).
3. Run the script, and it will process the images and save the results in the `Processed` directory.

```python
import os
import cv2
import numpy as np

def crop_and_rotate_images(input_folder, output_folder, crop_coords):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        try:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read {image_path}")
                continue

            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            x1, y1, x2, y2 = crop_coords
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rotated_image.shape[1], x2)
            y2 = min(rotated_image.shape[0], y2)
            roi = rotated_image[y1:y2, x1:x2]
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, roi)

            print(f"Processed: {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    input_folder = "4x_images"
    output_folder = "Processed"
    crop_coords = (0, 600, 3000, 2400)
    crop_and_rotate_images(input_folder, output_folder, crop_coords)
