import cv2
import os
import cv2 as cv
from PIL import Image
import numpy as np

# Function to get the images and labels for training
data_dir="F:\part1\project\data"
image_path="F:\part1\project\img"
def get_images_and_labels(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    images = []
    labels = []

    for image_path in image_paths:
        # Convert the image to grayscale
        img = Image.open(image_path).convert('L')
        # Convert to numpy array
        img_np = np.array(img, 'uint8')
        # Get the label from the directory name
        label = int(os.path.split(image_path)[-1].split(".")[1])

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            # Extract the face region
            face = img_np[y:y+h, x:x+w]
            images.append(face)
            labels.append(label)

    return images, labels

# Directory containing training images (each person's images in a separate folder)
train_data_dir = 'F:\part1\project\dataset'

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get training images and labels
images, labels = get_images_and_labels(train_data_dir)

# Train the recognizer
recognizer.train(images, np.array(labels))

# Save the trained model to a file
recognizer.save('lbph_model.yml')
