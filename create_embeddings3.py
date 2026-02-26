# import required libraries
import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import normalize
from deepface import DeepFace
import albumentations as A

# Load YOLO face model (here you can replace yolov8n.pt with yolov8n-face.pt but you need python 3.10 or 3.11)
yolo = YOLO("yolov8n-face.pt")

# Data augmentation
transform_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5)
])

dataset_path = "dataset"
embedding_dict = {}

# NEW EMBEDDING FUNCTION USING DEEPFACE
#here we can also use facenet which is more accurate and powerful but it will support only in python version 3.10 /3.11
def get_embedding(face_img):

    face_img = cv2.resize(face_img, (160, 160))

    # DeepFace expects RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    embedding = DeepFace.represent(
        img_path = face_img,
        model_name = "ArcFace",   # You can use Facenet, VGG-Face, ArcFace
        enforce_detection = False,
        detector_backend = "skip"  # Because YOLO already detected face
    )

    embedding_vector = np.array(embedding[0]["embedding"])
    return normalize([embedding_vector])[0]


# CREATE EMBEDDINGs
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)
    embeddings = []

    for img_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)

        results = yolo(img)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]

                # Apply augmentation
                augmented = transform_aug(image=face)["image"]

                emb = get_embedding(augmented)
                embeddings.append(emb)

    embedding_dict[person_name] = embeddings


# -------- SAVE EMBEDDINGS -------- #
os.makedirs("embeddings", exist_ok=True)

with open("embeddings/embeddings3.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)

print("Embeddings created successfully!")