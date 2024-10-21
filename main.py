import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Create a directory for saving images if it doesn't exist
if not os.path.exists("saved_images"):
    os.makedirs("saved_images")

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        name = name_entry.get()
        if name:
            image = Image.open(file_path)
            image.save(f"saved_images/{name}.jpg")  # Save image with the provided name
            messagebox.showinfo("Success", f"Image saved as {name}.jpg")
        else:
            messagebox.showwarning("Input Error", "Please enter a name for the image.")
    else:
        messagebox.showwarning("File Error", "No file selected.")

# Function to start webcam and recognize face
def start_webcam():
    known_face_path = f"saved_images/{name_entry.get()}.jpg"
    if not os.path.exists(known_face_path):
        messagebox.showwarning("File Error", "Please upload an image first.")
        return

    # Start the webcam
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Try to recognize the face in the current frame
        try:
            result = DeepFace.stream("saved_images")
            print(result)
        except Exception as e:
            print(e)

        # Display the frame in a window
        cv2.imshow("Webcam Feed", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Face Recognition App")

# Create and place widgets
tk.Label(root, text="Enter Name:").pack(pady=10)
name_entry = tk.Entry(root)
name_entry.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

start_button = tk.Button(root, text="Start Webcam", command=start_webcam)
start_button.pack(pady=10)

# Run the GUI
root.mainloop()
