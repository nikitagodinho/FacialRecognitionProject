import cv2
import os

def crop_faces(input_folder, output_folder):
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for image_name in os.listdir(input_folder):
        print(f"Processing file: {image_name}")

        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        # Read the image
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image {image_name}. Skipping...")
            continue

        # Convert the image to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection with modified parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Adjust for smaller face sizes
            minNeighbors=3,    # Increase sensitivity to detect more faces
            minSize=(20, 20)   # Reduce minimum size of detected faces
        )

        if len(faces) == 0:
            print(f"No faces detected in {image_name}. Check the image quality or alignment.")
        else:
            # Save each detected face
            for i, (x, y, w, h) in enumerate(faces):
                cropped_face = image[y:y+h, x:x+w]
                output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved cropped face to {output_path}")

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = "images/"  # Folder containing original images
    output_folder = "clipped_faces/"  # Folder to save cropped faces
    crop_faces(input_folder, output_folder)
