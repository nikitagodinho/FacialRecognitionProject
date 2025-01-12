import cv2
import os
import numpy as np
from deepface import DeepFace

# Class for cropping faces from images
class FaceCropper:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def crop_faces(self):
        for image_name in os.listdir(self.input_folder):
            image_path = os.path.join(self.input_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read {image_name}. Skipping...")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for i, (x, y, w, h) in enumerate(faces):
                cropped_face = image[y:y+h, x:x+w]
                output_path = os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved cropped face to {output_path}")

# Class for modifying images (pixelation, grayscale, noise)
class ImageModifier:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def pixelate_image(self, image, pixel_size=10):
        h, w = image.shape[:2]
        temp = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    def grayscale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def add_random_noise(self, image):
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        return cv2.add(image, noise)

    def modify_images(self):
        for image_name in os.listdir(self.input_folder):
            image_path = os.path.join(self.input_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read {image_name}. Skipping...")
                continue

            pixelated = self.pixelate_image(image, pixel_size=20)
            grayscale = self.grayscale_image(image)
            noisy = self.add_random_noise(image)

            cv2.imwrite(os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_pixelated.jpg"), pixelated)
            cv2.imwrite(os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_grayscale.jpg"), grayscale)
            cv2.imwrite(os.path.join(self.output_folder, f"{os.path.splitext(image_name)[0]}_noisy.jpg"), noisy)

# Class for detecting emotions from images
class EmotionRecognizer:
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def analyze_emotions(self):
        for image_name in os.listdir(self.input_folder):
            image_path = os.path.join(self.input_folder, image_name)
            print(f"Analyzing {image_name}...")
            try:
                result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                dominant_emotion = result['dominant_emotion']
                confidence = result['emotion'][dominant_emotion]
                print(f"Emotion: {dominant_emotion}, Confidence: {confidence:.2f}%")
            except Exception as e:
                print(f"Error analyzing {image_name}: {e}")

# Main program
if __name__ == "__main__":
    # Step 1: Crop Faces
    face_cropper = FaceCropper(input_folder='images/', output_folder='clipped_faces/')
    face_cropper.crop_faces()

    # Step 2: Modify Images
    modifier = ImageModifier(input_folder='clipped_faces/', output_folder='altered_images/')
    modifier.modify_images()

    # Step 3: Detect Emotions
    recognizer = EmotionRecognizer(input_folder='altered_images/')
    recognizer.analyze_emotions()
