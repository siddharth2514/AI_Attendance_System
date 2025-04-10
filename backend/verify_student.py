import cv2
import numpy as np
import mysql.connector
from deepface import DeepFace
from retinaface import RetinaFace
from scipy.spatial.distance import cosine
import face_alignment

# Initialize face alignment tool (using 2D landmarks on CPU)
fa = face_alignment.FaceAlignment(0, device='cpu', flip_input=False)

# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sidmus#25",
    database="attendance_system"
)
cursor = conn.cursor()

def adjust_lighting(image):
    """
    Boost the vibe of your image by adjusting lighting with histogram equalization.
    """
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return image_eq

def align_face(image):
    """
    Align the face using detected facial landmarks.
    """
    landmarks = fa.get_landmarks(image)
    if landmarks is None:
        print("No landmarks detected, returning original image.")
        return image
    pts = landmarks[0]
    # Calculate eye centers: left eye (points 36-41) and right eye (points 42-47)
    left_eye = np.mean(pts[36:42], axis=0)
    right_eye = np.mean(pts[42:48], axis=0)
    # Compute the angle between the eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    # Compute center between the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    # Get rotation matrix for the calculated angle and rotate the image
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_face

def preprocess_face(image):
    """
    Full preprocessing: adjust lighting and align the face.
    """
    image = adjust_lighting(image)
    image = align_face(image)
    return image

def fetch_student_embeddings():
    """
    Grab stored embeddings from MySQL and normalize 'em.
    """
    cursor.execute("SELECT registration_number, face_encoding FROM students")
    student_embeddings = {}
    for reg_no, encoding_blob in cursor.fetchall():
        if encoding_blob:
            embedding = np.frombuffer(encoding_blob, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            student_embeddings[reg_no] = embedding
    return student_embeddings

def verify_multiple_faces(image_path, threshold=0.60):
    """
    Detect multiple faces in one image using RetinaFace, verify each, and spit out results.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Error: Image at '{image_path}' not found!")
        return

    # Preprocess: adjust lighting and align the face
    img = preprocess_face(img)

    # Use RetinaFace to detect faces (returns a dict with keys for each face)
    detections = RetinaFace.detect_faces(img)
    if not detections:
        print("⚠️ No faces detected!")
        return

    student_embeddings = fetch_student_embeddings()
    verification_results = []

    # Iterate over each detected face; detections is a dict with each key being a face id
    for face_id, face_data in detections.items():
        # 'facial_area' is a list: [x1, y1, x2, y2]
        facial_area = face_data["facial_area"]
        x1, y1, x2, y2 = facial_area
        # Make sure coordinates are within bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        face_img = img[y1:y2, x1:x2]
        
        try:
            # Use DeepFace to represent the face with ArcFace; enforce_detection=False to avoid errors if detection is off
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
    image_path = "check.jpg"
    results = verify_multiple_faces(image_path)
    if results:
        for res in results:
            status = "Verified" if res["verified"] else "Not Verified"
            print(f"Face {res['face_id']}: {res['best_match']} with distance {res['distance']:.4f} -> {status}")
    
    cursor.close()
    conn.close()
