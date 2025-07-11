import csv
import psycopg2
import numpy as np

# PostgreSQL connection setup
conn = psycopg2.connect(
    host="",
    database="",
    user="",
    password="",
    port=5432
)
cursor = conn.cursor()

# Query data from students table (only existing columns)
cursor.execute("SELECT registration_number, face_encoding FROM students")
rows = cursor.fetchall()

# Write to CSV
with open("students_data_export.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["registration_number", "face_encoding"])
    
    for reg_no, encoding_bytes in rows:
        encoding = np.frombuffer(encoding_bytes, dtype=np.float32).tolist()
        writer.writerow([reg_no, encoding])

print("✅ Data exported successfully to 'students_data_export.csv'")

cursor.close()
conn.close()
