from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
import sqlite3
from datetime import datetime
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'attendance_secret_key'

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, student_id INTEGER, date TEXT,
                       FOREIGN KEY(student_id) REFERENCES students(id))''')
    conn.commit()
    conn.close()

init_db()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        name = request.form['student_name']
        if not name:
            flash('Name is required for capturing!', 'danger')
            return redirect(url_for('capture'))

        capture_faces(name)
        flash(f'Face images captured for {name}', 'success')
        return redirect(url_for('index'))
    return render_template('capture.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        train_model()
        flash('Model training completed successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        student_name = recognize_face()
        if student_name:
            mark_attendance(student_name)
            flash(f'Attendance marked for {student_name}', 'success')
        else:
            flash('Face not recognized!', 'danger')
    return render_template('recognize.html')

@app.route('/report')
def report():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT s.name, a.date FROM attendance a JOIN students s ON a.student_id = s.id''')
    records = cursor.fetchall()
    conn.close()
    return render_template('report.html', records=records)

# Utility Functions
def capture_faces(name):
    if not os.path.exists('data'):
        os.makedirs('data')

    student_path = os.path.join('data', name)
    os.makedirs(student_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        flash('Error: Could not access the webcam.', 'danger')
        return

    img_id = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            flash('Error: Failed to capture frame from webcam.', 'danger')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            img_id += 1
            file_name_path = os.path.join(student_path, f"{img_id}.jpg")
            cv2.imwrite(file_name_path, gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or img_id >= 50:
            break

    video_capture.release()
    cv2.destroyAllWindows()

def train_model():
    data_dir = "data"
    faces = []
    ids = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            try:
                if file.endswith(('jpg', 'png', 'jpeg')):
                    path = os.path.join(root, file)
                    img = Image.open(path).convert('L')
                    image_np = np.array(img, 'uint8')
                    folder_name = os.path.basename(root)
                    user_id = hash(folder_name) % 10000

                    faces.append(image_np)
                    ids.append(user_id)
                    print(f"Processed file: {path} with ID: {user_id}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    if len(faces) < 2:
        print("Not enough data to train the model. Add more face samples.")
        return

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.yml")
    print("Training completed and model saved as 'classifier.yml'.")

def recognize_face():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.yml")
    video_capture = cv2.VideoCapture(0)
    student_name = None
    confidence_threshold = 60  # Adjusted threshold for testing

    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            id, confidence = clf.predict(gray[y:y + h, x:x + w])
            print(f"Detected Face - ID: {id}, Confidence: {confidence}")

            if confidence < confidence_threshold:
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM students WHERE id = ?', (id,))
                result = cursor.fetchone()
                conn.close()

                if result:
                    student_name = result[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, student_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Recognized: {student_name} with Confidence: {confidence}")
                else:
                    print(f"No student found in database for ID: {id}")
            else:
                print(f"Face not recognized. Confidence: {confidence}")

        cv2.imshow('Recognizing Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or student_name:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return student_name


def mark_attendance(student_name):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM students WHERE name = ?', (student_name,))
    student = cursor.fetchone()

    if not student:
        cursor.execute('INSERT INTO students (name) VALUES (?)', (student_name,))
        student_id = cursor.lastrowid
    else:
        student_id = student[0]

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO attendance (student_id, date) VALUES (?, ?)', (student_id, date))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
