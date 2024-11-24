from deepface import DeepFace
import cv2
import os
import numpy as np

def preprocess_image(image_path):

    # Create a directory for preprocessed images if it doesn't exist
    preprocessed_dir = "preprocessed_images"
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Generate a filename for the preprocessed image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = os.path.join(preprocessed_dir, f"{base_name}_preprocessed.jpg")

    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)


    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, crop to the face region
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = equalized[y:y+h, x:x+w]
    else:
        face = equalized

    # Resize the image to a standard size
    resized = cv2.resize(face, (224, 224))
    normalized = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



    # Save the preprocessed image
    cv2.imwrite(preprocessed_path, (normalized).astype(np.uint8))

    return preprocessed_path
def test_face_recognition():
    # Path to the image you want to test
    test_image_path = "./MILLIE-BOBBY-BROWN-BRITISH-130622-default-Sq-GettyImages-1240647438.jpeg"

    # Path to the directory containing your database of known faces
    db_path = "saved_images"

    try:
        # Preprocess the test image and get the path to the saved preprocessed image
        preprocessed_path = preprocess_image(test_image_path)

        
        # Normalize the image just before recognition

        result = DeepFace.find(
            img_path=preprocessed_path,
            db_path=db_path, 
            enforce_detection=False
        )


        if len(result) > 0 and len(result[0]) > 0:
            recognized_name = os.path.splitext(os.path.basename(result[0]["identity"][0]))[0]
            print(f"Face recognized: {recognized_name}")
        else:
            print("Face not recognized")

        # Print additional details
        print("Recognition result:", result)
        print(f"Preprocessed image saved at: {preprocessed_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_face_recognition()

