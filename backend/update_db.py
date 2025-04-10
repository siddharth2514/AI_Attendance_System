import sqlite3

conn = sqlite3.connect("attendance.db")  # Connect to the database
cursor = conn.cursor()

# Add the face_embedding column if it doesn't exist
try:
    cursor.execute("ALTER TABLE students ADD COLUMN face_embedding BLOB;")
    print("✅ Column 'face_embedding' added successfully!")
except sqlite3.OperationalError:
    print("⚠️ Column 'face_embedding' already exists.")

conn.commit()
conn.close()
