import cv2
import os
import numpy as np

def modify_images(input_folder, output_folder):
    """
    Applies pixelation, grayscale, and noise to images in the input folder and saves them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        image = cv2.imread(image_path)

        # Pixelation
        small = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
        pixelated = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        pixelated_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_pixelated.jpg")
        cv2.imwrite(pixelated_path, pixelated)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_grayscale.jpg")
        cv2.imwrite(gray_path, gray)

        # Add Noise
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        noisy_image = cv2.add(image, noise)
        noisy_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_noisy.jpg")
        cv2.imwrite(noisy_path, noisy_image)

        print(f"Modified images saved for {image_name}")
