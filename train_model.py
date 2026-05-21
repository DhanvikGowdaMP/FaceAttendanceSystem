import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

usn_list = os.listdir(dataset_path)

for i, usn in enumerate(usn_list):
    path = os.path.join(dataset_path, usn)

    for image_name in os.listdir(path):
        img_path = os.path.join(path, image_name)

        img = Image.open(img_path).convert('L')
        img_np = np.array(img, 'uint8')

        faces.append(img_np)
        ids.append(i)

recognizer.train(faces, np.array(ids))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save(trainer_path)

print("Model trained successfully!")