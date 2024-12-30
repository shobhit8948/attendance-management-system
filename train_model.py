import cv2
import os
import numpy as np

def train_model(dataset_path='dataset', output_file='classifier.yml'):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_data = []
    ids = []

    # Loop through dataset folder
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                id = int(os.path.basename(root))  # Folder name is the student ID
                image = cv2.imread(path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    face_data.append(gray[y:y + h, x:x + w])
                    ids.append(id)

    # Train the model
    recognizer.train(face_data, np.array(ids))
    recognizer.write(output_file)
    print(f"Training complete. Model saved to {output_file}")

if __name__ == "__main__":
    train_model()
