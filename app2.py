import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from deepface import DeepFace

app = Flask(__name__)

# Load YOLO face detector
yolo = YOLO("yolov8n-face.pt")

# Load saved embeddings
with open("embeddings/embeddings2.pkl", "rb") as f:
    embedding_dict = pickle.load(f)


# -------- EMBEDDING FUNCTION -------- #
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


# -------- FACE RECOGNITION FUNCTION -------- #
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

    return best_match


# -------- VIDEO STREAM -------- #
def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

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
                            0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)