import cv2
import os
import numpy as np

# Define functions for modifications
def pixelate_image(image, pixel_size=10):
    """Applies pixelation to the image."""
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def grayscale_image(image):
    """Converts the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def add_random_noise(image):
    """Adds random noise to the image."""
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    return cv2.add(image, noise)

# Define input and output folders
input_folder = 'clipped_faces/'  # Folder containing cropped face images
output_folder = 'altered_images/'  # Folder for saving modified images

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read {image_name}. Skipping...")
        continue

    # Apply modifications
    pixelated = pixelate_image(image, pixel_size=20)
    grayscale = grayscale_image(image)
    noisy = add_random_noise(image)

    # Save the modified images
    cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_pixelated.jpg"), pixelated)
    cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_grayscale.jpg"), grayscale)
    cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_noisy.jpg"), noisy)

print("Image modifications completed. Check the 'altered_images/' folder.")
