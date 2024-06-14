import os
import cv2
import numpy as np

def crop_and_rotate_images(input_folder, output_folder, crop_coords):
    #Image Cropping function
    os.makedirs(output_folder, exist_ok=True)
    #Makes the output directory if not already made

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]  #Lists all images in th input dir

    for image_file in image_files: #Processes each image individually
        try:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read {image_path}")
                continue

            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #Image rotated by 90 degrees anti-clockwise


            x1, y1, x2, y2 = crop_coords
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rotated_image.shape[1], x2)
            y2 = min(rotated_image.shape[0], y2)
            roi = rotated_image[y1:y2, x1:x2] #Creates region of interest based on entered parameters
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, roi) #Image cropped and saved in the output directory

            print(f"Processed: {image_file}")

        except Exception as e:
            #Error message if the program is unable to read the file
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    input_folder = "4x_images"
    output_folder = "Processed"
    #Input and output directories
    crop_coords = (0, 600,3000, 2400)  #Crops images to this area (X-Cordinate, Y-Cordinate, Width, Height)
    crop_and_rotate_images(input_folder, output_folder, crop_coords)#Pass value to function