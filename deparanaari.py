import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image',img)
# cv2.waitKey(0)

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

def intuit():
    print("hi There")
intuit()
