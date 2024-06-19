import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import concurrent.futures
import tkinter as tk
import threading
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK
cred = credentials.Certificate("F:/AppKivy/madina-e8f8d-firebase-adminsdk-wnief-51540603d4.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://madina-e8f8d-default-rtdb.firebaseio.com'
})

# Initialize FaceNet
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes (1).npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model
model = pickle.load(open("svm_model_160x160 (1).pkl", 'rb'))
data = pd.read_excel('student_data.xlsx')

# Set confidence threshold
confidence_threshold = 20.3000

# Tkinter window
root = tk.Tk()
root.withdraw()

# Directories for screenshots and attendance
screenshot_dir = 'F:/part1/project/images'
os.makedirs(screenshot_dir, exist_ok=True)
attendance_directory = 'F:/part1/project/attendance'
os.makedirs(attendance_directory, exist_ok=True)

unknown_persons = set()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Log attendance
def log_attendance(name):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = os.path.join(attendance_directory, f'attendance_{current_date}.csv')
    current_time = now.strftime("%H:%M:%S")
    if not os.path.exists(attendance_file):
        attendance_data = pd.DataFrame({'Name': [name], 'Date': [current_date], 'Time': [current_time]})
        attendance_data.to_csv(attendance_file, index=False)
    else:
        df = pd.read_csv(attendance_file)
        if not df[(df['Name'] == name)].empty:
            return
        attendance_data = pd.DataFrame({'Name': [name], 'Date': [current_date], 'Time': [current_time]})
        attendance_data.to_csv(attendance_file, mode='a', header=False, index=False)

# Convert to PDF
def convert_to_pdf():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = os.path.join(attendance_directory, f'attendance_{current_date}.csv')
    df = pd.read_csv(attendance_file)
    plt.figure(figsize=(10, 6))
    plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.axis('off')
    plt.savefig(f'attendance_{current_date}.pdf')

# Process frame
def process_frame(frame):
    results = recognize_faces(frame)
    for (x, y, w, h, final_name) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, str(final_name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    return frame

# Recognize faces
def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        embeddings = facenet.embeddings(face_img)
        confidence = np.max(model.decision_function(embeddings))
        face_name = model.predict(embeddings)
        if confidence < confidence_threshold:
            final_name = "Unknown"
            unknown_persons.add(tuple(face_img[0].flatten()))
            save_screenshot(frame, x, y, w, h)
        else:
            final_name = encoder.inverse_transform(face_name)[0]
            log_attendance(final_name)
            update_firebase_status(final_name)
        results.append((x, y, w, h, final_name))
    return results

# Update Firebase status
def update_firebase_status(name):
    # Replace 'Dr.Mohamed Elnabawy' with the specific person's name you want to recognize
    if name == 'Dr.Mohamed Elnabawy':
        ref = db.reference('recognized_person')
        ref.set({'name': name, 'status': '1'})
    else:
        ref = db.reference('recognized_person')
        ref.set({'name': 'Unknown', 'status': '0'})

# Save screenshot
def save_screenshot(frame, x, y, w, h):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(screenshot_dir, f"unknown_{timestamp}.jpg")
    cv2.imwrite(file_name, frame[y:y+h, x:x+w])
    return file_name

# Run attendance
def run_attendance():
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=2)
    cap = cv2.VideoCapture(0)
    while datetime.now() < end_time:
        success, frame = cap.read()
        if success:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = executor.submit(process_frame, frame)
                frame = result.result()
            cv2.imshow("Object Tracking & Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# Monitor faces
def monitor_faces():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = executor.submit(process_frame, frame)
                frame = result.result()
            cv2.imshow("Object Tracking & Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    run_attendance()
    monitor_faces()
    convert_to_pdf()
