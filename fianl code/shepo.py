import cv2
from ultralytics import YOLO
import torch
import firebase_admin
from firebase_admin import credentials, db

# Load the YOLOv8 model
model = YOLO('best (3).pt')
model = YOLO('best (4).pt')
cred = credentials.Certificate("F:\AppKivy\madina-e8f8d-firebase-adminsdk-wnief-51540603d4.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://madina-e8f8d-default-rtdb.firebaseio.com/'
})
# Define class names based on your training
# Reference to the key status in Firebase
key_status_ref = db.reference('key_status')
class_names = ['knife','gun']
def update_key_status(status):
    key_status_ref.set(status)
# Initialize video capture (0 for default camera, or specify a video file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Initialize a flag to check if any weapon or cigarette is detected
    detected = False

    # Parse results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class ID and confidence score
            cls_id = int(box.cls[0].cpu().numpy())
            score = box.conf[0].cpu().numpy()

            # Draw bounding box and label on the frame
            label = f"{class_names[cls_id]}: {score:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Green for weapon, Red for cigarette
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If weapon or cigarette is detected, set the flag to True
            if class_names[cls_id] in ['knife','gun']:
                detected = True

    # Update Firebase key status based on detection
    if detected:
        update_key_status('1')
    else:
        update_key_status('0')

    # Display the frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()