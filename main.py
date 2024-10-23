import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from deepface import DeepFace
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")
        self.master.geometry("600x400")
        self.master.configure(bg="#f0f0f0")

        self.create_styles()
        self.create_widgets()
        self.result_window = None

    def create_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.bg_color = "#f0f0f0"
        self.accent_color = "#4a7abc"
        self.text_color = "#333333"

        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Calibri", 12))
        self.style.configure("TEntry", font=("Calibri", 12))
        self.style.configure("TButton", background=self.accent_color, foreground="white", font=("Calibri", 12, "bold"), padding=10)
        self.style.map("TButton", background=[('active', '#3a5d94')])

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Logo (replace 'logo.png' with your actual logo file)
        # logo_img = Image.open("logo.png").resize((100, 100), Image.LANCZOS)
        # self.logo = ImageTk.PhotoImage(logo_img)
        # logo_label = ttk.Label(main_frame, image=self.logo, background=self.bg_color)
        # logo_label.pack(pady=(0, 20))

        # Name entry frame
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))

        name_label = ttk.Label(name_frame, text="Enter Name:")
        name_label.pack(side=tk.LEFT, padx=(0, 10))

        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.upload_button = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.start_button = ttk.Button(button_frame, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # Instructions
        self.instructions = ttk.Label(main_frame, text="Upload an image or start webcam to recognize a face", 
                                      font=("Calibri", 10), foreground="gray")
        self.instructions.pack(pady=(10, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            name = self.name_entry.get()
            if name:
                image = Image.open(file_path)
                image.save(f"saved_images/{name}.jpg")  # Save image with the provided name
                messagebox.showinfo("Success", f"Image saved as {name}.jpg")
            else:
                messagebox.showwarning("Input Error", "Please enter a name for the image.")
        else:
            messagebox.showwarning("File Error", "No file selected.")

    def start_webcam(self):
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
                    self.recognize_face("snapshot.jpg")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()

        capture_and_recognize()

    def restart_webcam(self):
        if self.result_window:
            self.result_window.destroy()
        self.start_webcam()
    def recognize_face(self, image_path):
        try:
            result = DeepFace.find(image_path, db_path="saved_images", enforce_detection=False)

            if self.result_window:
                self.result_window.destroy()

            self.result_window = tk.Toplevel(self.master)
            self.result_window.title("Face Recognition Result")
            self.result_window.configure(bg=self.bg_color)

            img = Image.open(image_path)
            img = img.resize((600, 500), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            img_label = ttk.Label(self.result_window, image=img_tk, background=self.bg_color)
            img_label.image = img_tk
            img_label.pack(side="left", padx=10, pady=10)

            if len(result) > 0 and len(result[0]) > 0:
                recognized_name = os.path.splitext(os.path.basename(result[0]["identity"][0]))[0]
                result_label = ttk.Label(self.result_window, text=f"Face recognized: {recognized_name}", font=("Calibri", 16))
            else:
                result_label = ttk.Label(self.result_window, text="Face not recognized", font=("Calibri", 16))
                start_button = ttk.Button(self.result_window, text="Start Webcam", command=self.restart_webcam)
                start_button.pack(side="right", padx=10, pady=10)
            result_label.pack(side="right", padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during face recognition: {str(e)}")

    
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
