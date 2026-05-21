import cv2
import os
import pandas as pd
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

students = os.listdir("dataset")

cam = cv2.VideoCapture(0)

# Store marked students
marked_students = []

while True:

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:

            name = students[id]

            # Mark attendance only once
            if name not in marked_students:

                time_now = datetime.now().strftime("%H:%M:%S")
                marked_students.append([name, time_now])

                print("Attendance marked for:", name)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:

            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()


# Save attendance
df = pd.DataFrame(marked_students, columns=["USN","Time"])

if not os.path.exists("attendance"):
    os.makedirs("attendance")

file_name = "attendance/attendance_" + datetime.now().strftime("%Y%m%d") + ".xlsx"

df.to_excel(file_name, index=False)

print("Attendance saved in:", file_name)