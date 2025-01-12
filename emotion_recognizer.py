from deepface import DeepFace
import os

def analyze_emotions(input_folder, output_folder="emotion_results/"):
    """
    Analyzes emotions for all images in the input folder and saves results to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][dominant_emotion]

            result_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_result.txt")
            with open(result_path, "w") as f:
                f.write(f"Image: {image_name}\n")
                f.write(f"Dominant Emotion: {dominant_emotion}\n")
                f.write(f"Confidence: {confidence:.2f}%\n")

            print(f"Analyzed {image_name}: {dominant_emotion} ({confidence:.2f}%)")

        except Exception as e:
            print(f"Error analyzing {image_name}: {e}")
