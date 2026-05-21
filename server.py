from flask import Flask, render_template, Response
import cv2
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

students = os.listdir("dataset")

marked = set()
attendance = []

camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if confidence < 70:

                name = students[id]

                if name not in marked:

                    marked.add(name)

                    time_now = datetime.now().strftime("%H:%M:%S")

                    attendance.append([name,time_now])

                cv2.putText(frame,name,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            else:

                cv2.putText(frame,"Unknown",(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save')
def save():

    df = pd.DataFrame(attendance,columns=["USN","Time"])

    if not os.path.exists("attendance"):
        os.makedirs("attendance")

    file = "attendance/attendance_"+datetime.now().strftime("%Y%m%d")+".xlsx"

    df.to_excel(file,index=False)

    return "Attendance Saved"


app.run(host="0.0.0.0",port=5000)