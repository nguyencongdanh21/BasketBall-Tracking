#%%
# import numpy as np
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from cvzone import ColorFinder
from keras.utils.image_utils import img_to_array
from keras.utils import load_img
import time

state = 0
model = tf.keras.models.load_model('CNN_Bask_Rec.h5', compile=False)
cap = cv.VideoCapture('bongro1.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Create tracker
tracker = cv.TrackerCSRT_create()
# # Load 1st frame
ret, frame = cap.read()
mycolor = ColorFinder(False)
hsvVal = {'hmin': 4, 'smin': 100, 'vmin': 60, 'hmax': 17, 'smax': 255, 'vmax': 255}
framecount = 0
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frameblur = cv.GaussianBlur(frame, (5, 5), 1)
    imgColor, mask =mycolor.update(frameblur, hsvVal)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > 1500:
            #crop roi
            x,y,w,h=cv.boundingRect(cnt)
            roi=imgColor[y:y+h, x:x+w]
            #resize to 28x28
            roi=cv.resize(roi,(28,28))
            #roi = load_img(roi,target_size=(28,28))
            cv.imshow('Basket',roi)
            # normalize range(0,1)
            roi = img_to_array(roi)
            roi=roi.reshape(3,28,28,1)
            roi = roi.astype('float32')
            roi =roi/255
            #predict
            while state == 0:
                result=np.argmax(model.predict(roi),axis=1)
                if result[0] == 0:
                    ret = tracker.init(frame, (x, y, w, h))
                    state=1
                    if state == 1:
                        break
            ret, obj = tracker.update(frame)        
            if ret:
                if result[0] == 0:
                    p1 = (int(obj[0]),int(obj[1]))
                    p2 = (int(obj[0] + obj[2]),int(obj[1] + obj[3]))
                    cv.rectangle(frame, p1,p2, (255,0,0), 2, 1)
                else:
                    state = 0
                    break
            else:
                state = 0
    #framecount+=1
    #print(framecount)
    cv.imshow("Frame", frame)   
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
# %%

#%%
# import numpy as np
import cv2 as cv
import numpy as np
import os
from cvzone import ColorFinder



cap = cv.VideoCapture('basket.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Create tracker
#tracker = cv.TrackerCSRT_create()
# # Load 1st frame
ret, frame = cap.read()
# frame = cv.resize(frame,(320,240)) 

# detect object location
#r = cv.selectROI(frame)
# r = np.array([x,y,w,h])
# initialize the tracker
#ret = tracker.init(frame, r)

mycolor = ColorFinder(True)
hsvVal = 'red'

while True:

    img = cv.imread('frame70.jpg')
    imgColor, mask =mycolor.update(img, hsvVal)
        
    cv.imshow("img",img)
    cv.imshow("h",imgColor)
    #cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
#cap.release()
#cv.destroyAllWindows()
# %%
