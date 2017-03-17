import cv2
import numpy as np

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)

        comp = gray[y:y+h,x:x+w]
        cv2.imshow('frame', comp)
        comp = cv2.resize(comp, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.imshow('frame', comp)
        Id, conf = recognizer.predict(comp)
        print Id, conf
        if(conf<50):
            if(Id==1):
                Id="Thushar"
            elif(Id==2):
                Id="Modi"
            elif (Id == 3):
                Id = "Prabhu"
        else:
            Id=""
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()