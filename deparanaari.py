
import cv2 as cv2
#import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt



def templateMatching():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        template = cv2.imread('watch_crop.jpg', 0)
        h,w = template.shape[::]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

        cv2.imshow('Detected', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


templateMatching()


def edgeDetectionAndGradient():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        laplacian = cv2.Laplacian(frame,cv2.CV_64F)
        sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.Canny(frame,200,200)

        cv2.imshow('frame', frame)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('sobely', sobely)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('edges', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



#edgeDetectionAndGradient()


def filter_video():
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower_red = np.array([150,150,50])
        upper_red = np.array([180,255,150])

        mask = cv2.inRange(hsv,lower_red,upper_red)
        res = cv2.bitwise_and(frame,frame,mask = mask)

        kernel = np.ones((15,15),np.float32)/225
        smoothed = cv2.filter2D(res,-1,kernel)
        blur = cv2.GaussianBlur(res, (15,15),0)
        blur = cv2.medianBlur(res, 15)
        blur = cv2.bilateralFilter(res, 15,75,75)


        opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('frame', frame)
        #cv2.imshow('mask', mask)
        #cv2.imshow('res', res)
        #cv2.imshow('smoothed', smoothed)
        #cv2.imshow('blur', blur)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#filter_video()


def theshholding():
    img = cv2.imread('bookpage.jpg')
    retval,threshold = cv2.threshold(img,12,255,cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval2,threshold2 = cv2.threshold(gray,12,255,cv2.THRESH_BINARY)

    gauss = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

    retval,otsu = cv2.threshold(gauss,125,255,cv2.THRESH_OTSU)

    cv2.imshow('image', img)
    cv2.imshow('threshhold', threshold)
    cv2.imshow('threshhold2', threshold2)
    cv2.imshow('gauss', gauss)
    cv2.imshow('otsu', otsu)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

def imageOperations():
    img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
    img[55,55] =[0,0,255]

    roi = img[100:150,100:150]  = [255,255,255]
    watch_face = img[37:111,107:194]
    img[0:74,0:87] = watch_face



    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#scan()
#markWatchImage()
#imageOperations()
#thresholding()





