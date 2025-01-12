import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from face_cropper import crop_faces
from image_modifier import modify_images
from emotion_recognizer import analyze_emotions

# Global variables to store the uploaded image path
uploaded_image_path = None

def upload_image():
    global uploaded_image_path, img_label

    # Open file dialog to select an image
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not uploaded_image_path:
        return

    # Display the uploaded image
    img = Image.open(uploaded_image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)

    # Update the image label
    img_label.configure(image=img)
    img_label.image = img

def process_image():
    global uploaded_image_path

    if not uploaded_image_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return

    try:
        # Create required directories
        cropped_folder = "clipped_faces/"
        modified_folder = "altered_images/"
        results_folder = "emotion_results/"

        if not os.path.exists(cropped_folder):
            os.makedirs(cropped_folder)
        if not os.path.exists(modified_folder):
            os.makedirs(modified_folder)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Step 1: Crop faces
        crop_faces(os.path.dirname(uploaded_image_path), cropped_folder)

        # Step 2: Modify images
        modify_images(cropped_folder, modified_folder)

        # Step 3: Analyze emotions
        analyze_emotions(modified_folder, results_folder)

        # Read and display results for the first processed image
        result_files = [f for f in os.listdir(results_folder) if f.endswith(".txt")]
        if result_files:
            with open(os.path.join(results_folder, result_files[0]), "r") as f:
                results_text = f.read()
        else:
            results_text = "No emotion results found."

        # Display the results in a popup
        messagebox.showinfo("Emotion Analysis Results", results_text)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main GUI window
root = tk.Tk()
root.title("Facial Recognition and Emotion Detection")

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Image display
img_label = tk.Label(root)
img_label.pack()

# Process button
process_button = tk.Button(root, text="Process Image", command=process_image)
process_button.pack()

# Start the GUI event loop
root.mainloop()
