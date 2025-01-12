from deepface import DeepFace
import os

def analyze_emotions(input_folder, output_folder="emotion_results/"):
    """
    Analyzes emotions for all images in the input folder and saves results to the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing images to analyze.
    - output_folder (str): Path to the folder where results will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Process each file in the input folder
    for image_name in os.listdir(input_folder):
        print(f"Analyzing {image_name}...")

        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        image_path = os.path.join(input_folder, image_name)

        try:
            # Analyze the emotion using DeepFace
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

            # Handle DeepFace result structure
            if isinstance(result, list):
                result = result[0]  # Access the first result if it's a list
            
            # Extract the dominant emotion and confidence
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
            emotion_confidences = result.get('emotion', {})
            confidence = emotion_confidences.get(dominant_emotion, 0)

            # Save the results to a text file
            result_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_result.txt")
            with open(result_path, "w") as f:
                f.write(f"Image: {image_name}\n")
                f.write(f"Dominant Emotion: {dominant_emotion}\n")
                f.write(f"Confidence: {confidence:.2f}%\n")

            print(f"Emotion: {dominant_emotion}, Confidence: {confidence:.2f}% - Results saved to {result_path}")

        except Exception as e:
            print(f"Error analyzing {image_name}: {e}")

if __name__ == "__main__":
    # Specify the input folder and output folder
    input_folder = "altered_images/"  # Folder with modified images
    output_folder = "emotion_results/"  # Folder to save emotion analysis results

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Run the analysis
    analyze_emotions(input_folder, output_folder)

