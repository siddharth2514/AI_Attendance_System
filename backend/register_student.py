import cv2
import numpy as np
import mysql.connector
from deepface import DeepFace
from verify_student import preprocess_face  # Reusing the same preprocessing

# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sidmus#25",
    database="attendance_system"
)
cursor = conn.cursor()

def register_student_multi(registration_number, name, image_paths):
    embeddings = []

    # Process each image
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Error: Image at '{image_path}' not found!")
            continue

        # Full preprocessing (adjust lighting + align face)
        img = preprocess_face(img)

        try:
            # Get face embedding using ArcFace model from DeepFace
            face_repr = DeepFace.represent(img, model_name="ArcFace")[0]["embedding"]
        except Exception as e:
            print(f"⚠️ Face not detected in image '{image_path}': {e}")
            continue

        # Convert embedding to numpy array and normalize it
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

    # Average the embeddings and normalize the final embedding
    avg_embedding = np.mean(embeddings, axis=0)
    avg_norm = np.linalg.norm(avg_embedding)
    if avg_norm > 0:
        avg_embedding = avg_embedding / avg_norm
    else:
        print("⚠️ Averaged embedding has zero norm!")
        return

    # Convert the averaged embedding to bytes for MySQL storage
    face_encoding_blob = avg_embedding.tobytes()

    # Insert into MySQL
    try:
        cursor.execute(
            "INSERT INTO students (registration_number, name, face_encoding) VALUES (%s, %s, %s)",
            (registration_number, name, face_encoding_blob)
        )
        conn.commit()
        print(f"✅ Student {name} registered successfully!")
    except mysql.connector.Error as err:
        print(f"⚠️ MySQL Error: {err}")

if __name__ == "__main__":
    # List of images for registration
    image_paths = [
        "reg2.jpg",
        "reg2(2).jpg",
        "reg2(3).jpg",
        "reg2(4).jpg",
        "reg2(5).jpg"
        
    ]
    register_student_multi("RA2211042010002", "Cevin", image_paths)

    cursor.close()
    conn.close()
