
import cv2 as cv2
#import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt




# cv2.imshow('image',img)
# cv2.waitKey(0)
def scan():
    img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Color', frame)
        cv2.imshow('Grey',gray)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def markWatchImage():
    img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
    #line
    cv2.line(img,(0,0),(150,150),(255,255,255),15)
    #rectangles
    cv2.rectangle(img,(15,15),(150,150),(0,255,0),5)
    #circle
    cv2.circle(img,(100,62),55,(0,0,255),4)
    #poligon
    pts = np.array([[10,5],[4,7],[180,200]],np.int32)
    cv2.polylines(img,[pts],True,(0,255,255),3)

    #Text
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,'Thushar',(0,180),font,0.5,(0,0,255))

    #show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def intuit():
    print("hi There")
intuit()
markWatchImage()
