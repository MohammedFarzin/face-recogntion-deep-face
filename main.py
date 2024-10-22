import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from deepface import DeepFace
from PIL import Image, ImageTk


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
def start_webcam():
    video_capture = cv2.VideoCapture(0)
    
    def capture_and_recognize():
        face_cascade = None
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load face detection classifier: {str(e)}")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Webcam Feed", frame)

            if len(faces) > 0:
                cv2.imwrite("snapshot.jpg", frame)
                video_capture.release()
                cv2.destroyAllWindows()
                recognize_face("snapshot.jpg")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def recognize_face(image_path):
        try:
            result = DeepFace.find(image_path, db_path="saved_images", enforce_detection=False)
            
            result_window = tk.Toplevel(root)
            result_window.title("Face Recognition Result")
            
            img = Image.open(image_path)
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(result_window, image=img_tk)
            img_label.image = img_tk
            img_label.pack(side="left", padx=10, pady=10)
            
            if len(result) > 0 and len(result[0]) > 0:
                recognized_name = os.path.splitext(os.path.basename(result[0]["identity"][0]))[0]
                result_label = tk.Label(result_window, text=f"Face recognized: {recognized_name}", font=("Arial", 16))
            else:
                result_label = tk.Label(result_window, text="Face not recognized", font=("Arial", 16))
                start_button = tk.Button(result_window, text="Start Webcam", command=start_webcam)
                start_button.pack(pady=10)
            result_label.pack(side="right", padx=10, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during face recognition: {str(e)}")

    capture_and_recognize()

# Create the main window
root = tk.Tk()
root.geometry("800x600")
root.minsize(400, 300)
root.maxsize(1200, 900)
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