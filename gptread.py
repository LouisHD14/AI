import os
import cv2 as cv


face_cascade = cv.CascadeClassifier('C:\\Users\\Louis_HD\\Desktop\\dev\\test\\models\\haar_face.xml')


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray People', gray)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)


    print(f'Number of faces found = {len(faces)}')
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)


    cv.imshow('Detected Faces', frame)


    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
