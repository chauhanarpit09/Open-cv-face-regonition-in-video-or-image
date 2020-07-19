
# %%
import cv2 as cv
import numpy as np

#%%
cap = cv.VideoCapture(0)
face_clf = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_clf.detectMultiScale(gray,1.1,4)
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv.imshow('img',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()

# %%
face_clf = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('img.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces = face_clf.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

cv.imshow('img',img)
cv.waitKey(1)

# %%
