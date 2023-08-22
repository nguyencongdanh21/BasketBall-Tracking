#%%
import cv2 as cv
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Load model
model = tf.keras.models.load_model('CNN_Bask_Rec.h5', compile=False)
print("Import model complete!")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create tracker
tracker = cv.legacy.TrackerMOSSE_create()
ret, frame = cap.read()

tracking = 0
x, y, w, h = 0, 0, 0, 0

def my_basketball(frame):
    blur = cv.GaussianBlur(frame, (3, 3), 1)
    # Convert Color to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    b_img = cv.inRange(hsv, (4, 125, 60), (17, 255, 255))
    contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw Bounding Box
    for cnt in contours:
        global x, y, w, h , tracking

        if cv.contourArea(cnt) > 1500:
            x, y, w, h = cv.boundingRect(cnt)
            # Crop Based On Bounding Box
            crop_img = frame[y:y+h, x:x+w]
            # Bring to CNN model
            # Convert Color To Gray
            gray_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

            # Resize image to (28, 28)
            img_resized = cv.resize(gray_img, (28, 28))
            cv.imshow("Resize",img_resized)
            # Normalize image
            img_resized = img_resized.astype(float) / 255.0
            img_reshaped = np.reshape(img_resized, (1, 28, 28, 1))
            result = np.argmax(model.predict(img_reshaped))
            print(result)
            # If detection is successful
            if result == 1:    # Class 1 corresponds to "BasketBall"
                tracking = 1
                print(x, y, w, h)
                print("Tracking Status",tracking)
                print("Prict",result)
                return tracking, (x, y, w, h)

    tracking = 0
    print("Tracking Status",tracking)
    return tracking, (x, y, w, h)

# Read frames from camera in a while loop
frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Resize to 720p resolution
    frame = cv.resize(frame, (1280, 720))

    # Trong vòng lặp while
    if tracking == 0:
        tracking, bbox = my_basketball(frame)
        if tracking == 1:
            ret = tracker.init(frame, bbox)
            print("Tracking Status", tracking)
    else:
        ret, obj = tracker.update(frame)
        if ret:
            p1 = (int(obj[0]), int(obj[1]))
            p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
            cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            p1 = (0, 0)
            p2 = (0, 0)
            print("Tracking failed")
            tracking = 0
            print("Tracking Status", tracking)
            
    cv.imshow("First Frame", frame)

    # Stop Button
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
# %%
