import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

usn = input("Enter Student USN: ")

dataset_path = "dataset/" + usn

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam=cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]

        file_name = dataset_path + "/img" + str(count) + ".jpg"

        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capturing Faces", frame)

    if count >= 30:
        break

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

print("Face dataset created successfully!")