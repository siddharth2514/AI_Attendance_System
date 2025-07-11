import os
import cv2
import numpy as np
import psycopg2
from deepface import DeepFace
import face_alignment

# Initialize face alignment tool
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

# PostgreSQL connection setup
conn = psycopg2.connect(
    host="localhost",
    database="attendance_system",
    user="postgres",
    password="admin",
    port=5432
)
cursor = conn.cursor()

def adjust_lighting(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return image_eq

def align_face(image):
    landmarks = fa.get_landmarks(image)
    if landmarks is None:
        print("No landmarks detected, returning original image.")
        return image
    pts = landmarks[0]
    left_eye = np.mean(pts[36:42], axis=0)
    right_eye = np.mean(pts[42:48], axis=0)
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_face

def preprocess_face(image):
    image = adjust_lighting(image)
    image = align_face(image)
    return image

def register_student(registration_number, name, folder_path):
    embeddings = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Error reading image: {image_path}")
                continue

            img = preprocess_face(img)

            try:
                face_repr = DeepFace.represent(img, model_name="ArcFace")[0]["embedding"]
            except Exception as e:
                print(f"⚠️ Face not detected in image '{image_path}': {e}")
                continue

            embedding = np.array(face_repr, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"⚠️ Zero norm for embedding from image '{image_path}'!")
                continue
            normalized_embedding = embedding / norm
            embeddings.append(normalized_embedding)

    if not embeddings:
        print("⚠️ No valid images found for registration!")
        return

    avg_embedding = np.mean(embeddings, axis=0)
    avg_norm = np.linalg.norm(avg_embedding)
    if avg_norm > 0:
        avg_embedding = avg_embedding / avg_norm
    else:
        print("⚠️ Averaged embedding has zero norm!")
        return

    face_encoding_blob = avg_embedding.tobytes()

    try:
        cursor.execute(
            "INSERT INTO students (registration_number, name, face_encoding) VALUES (%s, %s, %s)",
            (registration_number, name, psycopg2.Binary(face_encoding_blob))
        )
        conn.commit()
        print(f"✅ Student {name} registered successfully!")
    except psycopg2.Error as err:
        print(f"⚠️ PostgreSQL Error: {err}")

if __name__ == "__main__":
    register_student(
        "RA2211042010003",
        "Aryan",
        r"C:\Users\annan\Desktop\AI_Attendance_System\student_faces_db\Reg 3"
    )

    cursor.close()
    conn.close()
