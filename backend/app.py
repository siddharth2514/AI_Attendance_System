from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import os
from retinaface import RetinaFace
from deepface import DeepFace
import mysql.connector
import cv2
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__, template_folder='.')
app.secret_key = "your_secret_key"  # CHANGE THIS TO SOMETHING SECURE!
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# MySQL connection setup (update credentials as needed)
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Sidmus#25",
        database="attendance_system"
    )
    cursor = conn.cursor()
    print("Connected to MySQL successfully!")
except mysql.connector.Error as err:
    print("Error connecting to MySQL:", err)

# ---------------- Helper Functions ---------------- #

def adjust_lighting(image):
    try:
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb_img)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        return image_eq
    except Exception as e:
        print("Error in adjust_lighting:", e)
        return image

def fetch_student_embeddings():
    student_embeddings = {}
    try:
        cursor.execute("SELECT registration_number, face_encoding FROM students")
        rows = cursor.fetchall()
        print(f"Fetched {len(rows)} student embeddings from DB")
        for reg_no, encoding_blob in rows:
            if encoding_blob:
                embedding = np.frombuffer(encoding_blob, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                student_embeddings[reg_no] = embedding
    except mysql.connector.Error as err:
        print("Error fetching student embeddings:", err)
    return student_embeddings

def verify_multiple_faces(image_path, threshold=0.4):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Could not read image from", image_path)
            return None

        print("Image read successfully. Adjusting lighting...")
        img = adjust_lighting(img)

        print("Detecting faces using RetinaFace...")
        detections = RetinaFace.detect_faces(img)
        if not detections:
            print("No faces detected in the image.")
            return None

        student_embeddings = fetch_student_embeddings()
        verification_results = []

        for face_id, face_data in detections.items():
            facial_area = face_data["facial_area"]
            x1, y1, x2, y2 = facial_area
            x1, y1 = max(x1, 0), max(y1, 0)
            face_img = img[y1:y2, x1:x2]
            print(f"Processing face {face_id} with area {facial_area}...")

            try:
                # Set enforce_detection=False so that even if the face isn't perfect, we still try to represent it
                face_repr = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
            except Exception as e:
                print(f"Error representing face {face_id}: {e}")
                continue

            embedding = np.array(face_repr, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"Face {face_id} has zero norm, skipping!")
                continue
            embedding = embedding / norm

            best_match = None
            best_score = float("inf")
            for reg_no, stored_embedding in student_embeddings.items():
                distance = cosine(embedding, stored_embedding)
                if distance < best_score:
                    best_score = distance
                    best_match = reg_no

            print(f"Face {face_id} best match: {best_match} with distance {best_score:.4f}")
            verification_results.append({
                "face_id": face_id,
                "best_match": best_match,
                "distance": best_score,
                "verified": best_score < threshold
            })

        return verification_results

    except Exception as e:
        print("Error in verify_multiple_faces:", e)
        return None

def get_attendance_details():
    try:
        cursor.execute("SELECT * FROM attendance")
        details = cursor.fetchall()
        print(f"Fetched {len(details)} attendance records.")
        return details
    except mysql.connector.Error as err:
        print("Error fetching attendance details:", err)
        return None

# ---------------- Routes ---------------- #

@app.route('/')
def home():
    print("Landing page accessed.")
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("Login route accessed.")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print("Login attempt for email:", email)

        try:
            cursor.execute("SELECT password FROM teachers WHERE email=%s", (email,))
            result = cursor.fetchone()
            if result and check_password_hash(result[0], password):
                session['user'] = email
                print("Login successful for", email)
                return redirect(url_for('upload'))
            else:
                error = "Invalid email or password"
                print("Login failed for", email)
                return render_template('login.html', error=error)
        except mysql.connector.Error as err:
            print("Database error during login:", err)
            return render_template('login.html', error="Database error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    print("Registration route accessed.")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print("Registration attempt for email:", email)

        if not email.endswith('@srmist.edu.in'):
            error = "Email must end with @srmist.edu.in"
            print("Invalid email domain for", email)
            return render_template('register.html', error=error)
        
        hashed_password = generate_password_hash(password)
        try:
            cursor.execute("INSERT INTO teachers (email, password) VALUES (%s, %s)", (email, hashed_password))
            conn.commit()
            print("Registration successful for", email)
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            error = f"Database error: {err}"
            print("Registration DB error for", email, ":", err)
            return render_template('register.html', error=error)
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        print("Unauthorized access to upload page.")
        return redirect(url_for('login'))
    print("Upload page accessed by", session['user'])
    if request.method == 'POST':
        if 'photo' not in request.files:
            print("No photo part in the request.")
            return redirect(request.url)
        file = request.files['photo']
        if file.filename == '':
            print("No file selected for uploading.")
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print("File saved to", file_path)
            results = verify_multiple_faces(file_path)
            return render_template('result.html', results=results)
    return render_template('upload.html')

@app.route('/attendance')
def attendance():
    if 'user' not in session:
        print("Unauthorized access to attendance page.")
        return redirect(url_for('login'))
    details = get_attendance_details()
    return render_template('attendance.html', details=details)

@app.route('/logout')
def logout():
    user = session.pop('user', None)
    print("User logged out:", user)
    return redirect(url_for('home'))

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
