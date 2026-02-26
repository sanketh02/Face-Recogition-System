import streamlit as st
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from deepface import DeepFace
from PIL import Image

# Load Models & Embeddings
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n-face.pt")

    with open("embeddings/embeddings3.pkl", "rb") as f:
        embedding_dict = pickle.load(f)

    return yolo, embedding_dict


yolo, embedding_dict = load_models()


# Embedding Function
def get_embedding(face_img):

    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    embedding = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="skip"
    )

    embedding_vector = np.array(embedding[0]["embedding"])
    return normalize([embedding_vector])


# Recognition Function
def recognize_face(face_embedding, threshold=0.6):

    best_match = "UNKNOWN"
    max_sim = 0

    for name, embeddings in embedding_dict.items():
        for emb in embeddings:
            sim = cosine_similarity(face_embedding, [emb])[0][0]

            if sim > max_sim:
                max_sim = sim
                best_match = name

    if max_sim < threshold:
        return "UNKNOWN"

    return f"{best_match} ({max_sim:.2f})"


# Streamlit UI
st.title("🔍 Face Recognition System ")

option = st.radio("Select Input Type:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = yolo(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                try:
                    emb = get_embedding(face)
                    name = recognize_face(emb)
                except:
                    name = "ERROR"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Result", use_column_width=True)


elif option == "Use Webcam":

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:

        image = Image.open(img_file_buffer)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = yolo(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                try:
                    emb = get_embedding(face)
                    name = recognize_face(emb)
                except:
                    name = "ERROR"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Webcam Result", use_column_width=True)
