import cv2
import os

def crop_faces(input_folder, output_folder):
    """
    Detects and crops faces from images in the input folder and saves them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
            cv2.imwrite(output_path, face)
            print(f"Saved cropped face to {output_path}")
