import cv2
import numpy as np
import psycopg2
from deepface import DeepFace
from retinaface import RetinaFace
from scipy.spatial.distance import cosine

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

def preprocess_face(image):
    # Just adjust lighting (no alignment)
    image = adjust_lighting(image)
    return image

def fetch_student_embeddings():
    cursor.execute("SELECT registration_number, face_encoding FROM students")
    student_embeddings = {}
    for reg_no, encoding_bytes in cursor.fetchall():
        if encoding_bytes:
            embedding = np.frombuffer(encoding_bytes, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            student_embeddings[reg_no] = embedding
    return student_embeddings

def verify_image(image_path, threshold=0.65):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Error: Image at '{image_path}' not found!")
        return

    img = preprocess_face(img)  # Adjust lighting, no alignment
    detections = RetinaFace.detect_faces(img)
    if not detections:
        print("⚠️ No faces detected!")
        return

    student_embeddings = fetch_student_embeddings()
    verification_results = []

    for face_id, face_data in detections.items():
        facial_area = face_data["facial_area"]
        x1, y1, x2, y2 = facial_area
        x1, y1 = max(x1, 0), max(y1, 0)
        face_img = img[y1:y2, x1:x2]
        
        try:
            face_repr = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"⚠️ Face detection error on face {face_id}: {e}")
            continue

        embedding = np.array(face_repr, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            print(f"⚠️ Zero norm for face {face_id}, skipping!")
            continue
        embedding = embedding / norm

        best_match = None
        best_score = float("inf")
        for reg_no, stored_embedding in student_embeddings.items():
            distance = cosine(embedding, stored_embedding)
            if distance < best_score:
                best_score = distance
                best_match = reg_no

        verification_results.append({
            "face_id": face_id,
            "best_match": best_match,
            "distance": best_score,
            "verified": best_score < threshold
        })

    return verification_results

if __name__ == "__main__":
    image_path = r"C:\\Users\\annan\\Desktop\\AI_Attendance_System\\Test_faces\\Test5\\class5.jpeg"
    results = verify_image(image_path)
    if results:
        for res in results:
            status = "Verified" if res["verified"] else "Not Verified"
            tick = "✅" if res["verified"] else "❌"  # Add green tick if verified, red cross if not
            print(f"Face {res['face_id']}: {res['best_match']} with distance {res['distance']:.4f} -> {tick} {status}")
    
    cursor.close()
    conn.close()

