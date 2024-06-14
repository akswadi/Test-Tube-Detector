import os
import cv2
import numpy as np
import pandas as pd
#importing all the required libraries

def init(input_dir,combdir):
    #Gets list of all jpg and jpeg files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg'))]
    for image_file in image_files:
        try:
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                #Converts image to grayscale for processing
                result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                comb_img = result_image
                comb_image_path = os.path.join(combdir, image_file)
                cv2.imwrite(comb_image_path, comb_img)
                #Contructing and writing image to output directory
            else:
                print(f"Error: Unable to read {image_path}")

        except Exception as e:
            #error messages incase the program fails to read the message
            print(f"Error processing {image_file}: {e}")


def image_process(input_dir, output_dir, results_file, combdir,roi,max_thresh,cir_thresh):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg'))]
    res_files = [f for f in os.listdir(combdir) if f.endswith(('.jpg', '.jpeg'))]
    x1,x2,y1,y2=roi #Unpack the roi values

    results = {}

    for image_file in image_files:
        try:
            image_path = os.path.join(input_dir, image_file)

            
            res_filename = image_file 
            res_path = os.path.join(combdir, res_filename)
            
            if not os.path.isfile(res_path):
                print(f"Error: Result file '{res_filename}' not found in directory '{combdir}'")
                continue
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            comb_img = cv2.imread(res_path)

            if image is not None and comb_img is not None:
                inverted_image = cv2.bitwise_not(image)
                inverted_image = cv2.GaussianBlur(inverted_image, (15, 15), 0)
                #Invert image and apply gaussian blur 

                x, w, y, h = x1,x2,y1,y2
                roi = inverted_image[y:y+h, x:x+w]

                # Apply binary thresholding
                _, binary_roi = cv2.threshold(roi, 143, 255, cv2.THRESH_BINARY)

                # Create a mask and apply the binary ROI
                mask = np.zeros_like(image, dtype=np.uint8)
                mask[y:y+h, x:x+w] = binary_roi

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Define thresholds for area and circularity
                min_area_threshold = 2900
                max_area_threshold = max_thresh
                circularity_threshold = cir_thresh

                # Convert the image to BGR (3-channel) for drawing
                result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # Draw the ROI rectangle on the images
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 225), 5)
                cv2.rectangle(comb_img, (x, y), (x+w, y+h), (255, 0, 225), 5)

                # Divide the ROI into sections and draw the division lines
                num_divisions = 8
                division_width = w // num_divisions
                for i in range(1, num_divisions):
                    cv2.line(result_image, (x + i * division_width, y), (x + i * division_width, y + h), (0, 200, 0), 5)
                    cv2.line(comb_img, (x + i * division_width, y), (x + i * division_width, y + h), (0, 200, 0), 5)

                division_results = {}
                for i in range(num_divisions):
                    division_mask = mask[:, x + i * division_width:x + (i + 1) * division_width]

                    division_contours, _ = cv2.findContours(division_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in division_contours:
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter == 0:
                            continue

                        circularity = (4 * np.pi * area) / (perimeter * perimeter)

                        if area >= min_area_threshold and area <= max_area_threshold and circularity >= circularity_threshold:
                            division_results[i] = "E"
                            cv2.putText(result_image, "E", (x + i * division_width + 10, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 6)
                            cv2.putText(result_image, str(i), (x + i * division_width + 45, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 6)
                            cv2.rectangle(result_image, (x + i * division_width, y), 
                                          (x + (i + 1) * division_width, y + h), (0,0,255), -1)
                            cv2.rectangle(comb_img, (x + i * division_width, y), 
                                          (x + (i + 1) * division_width, y + h), (0,0,255), -1)
                            cv2.putText(comb_img, "E", (x + i * division_width + 10, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 6)
                            cv2.putText(comb_img, str(i), (x + i * division_width + 45, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 6)
                            cv2.line(result_image, (x + i * division_width, y), (x + i * division_width, y + h), (0, 0, 0), 5)
                            cv2.line(comb_img, (x + i * division_width, y), (x + i * division_width, y + h), (0, 0, 0), 5)
                            break
                    else:
                        division_results[i] = "NE"
                        cv2.putText(result_image, "NE", (x + i * division_width + 10, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 6)
                        cv2.putText(result_image, str(i), (x + i * division_width + 52, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 6)
                        cv2.rectangle(result_image, (x + i * division_width, y), 
                                      (x + (i + 1) * division_width, y + h), (0,225,0), -1)
                        cv2.rectangle(comb_img, (x + i * division_width, y), 
                                      (x + (i + 1) * division_width, y + h), (0,225,0), -1)
                        cv2.putText(comb_img, "NE", (x + i * division_width + 10, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 6)
                        cv2.putText(comb_img, str(i), (x + i * division_width + 52, y + 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 6)
                        cv2.line(result_image, (x + i * division_width, y), (x + i * division_width, y + h), (0, 0, 0), 5)
                        cv2.line(comb_img, (x + i * division_width, y), (x + i * division_width, y + h), (0, 0, 0), 5)

                results[image_file] = division_results

                # Save the processed images
                output_image_path = os.path.join(output_dir, image_file)
                res_image_path = os.path.join(combdir, image_file)
                cv2.imwrite(output_image_path, result_image)
                cv2.imwrite(res_image_path, comb_img)

            else:
                print(f"Error: Unable to read or process {image_path} or {res_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    with open(results_file, 'w') as file:
        for image_file, division_results in results.items():
            file.write(f"Image: {image_file}\n")
            for division_index, status in division_results.items():
                file.write(f"Division {division_index}: {status}\n")
            file.write("\n")
            #Write the results to a text document

    print(f"Results saved to: {results_file}")


input_directory = "Processed"
output_directory = "Area_1"
output_directory2 = "Area_2"
output_directory3 = "Area_3"
output_directory4 = "Area_4"
comb_img_dir="Combdir"
results_file = "results.txt"
results_file2 = "results2.txt"
results_file3 = "results3.txt"
results_file4 = "results4.txt"


#Give separate roi cords
init(input_directory,comb_img_dir)
a1 = [20,1320,1170,180]
a2 = [20,1320,1020,155]
a3 = [1620,1220,30,155]
a4 = [1650,1220,860,155]

image_process(input_directory, output_directory, results_file,comb_img_dir,a1,7100,0.718)
image_process(input_directory, output_directory2, results_file2,comb_img_dir,a2,7100,0.718)
image_process(input_directory, output_directory3, results_file3,comb_img_dir,a3,7100,0.718)
image_process(input_directory, output_directory4, results_file4,comb_img_dir,a4,19000,0)