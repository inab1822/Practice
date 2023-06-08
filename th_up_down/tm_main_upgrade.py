import tensorflow
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time

# Load the model
model = load_model("model/Thumb_up_down/keras_model.h5", compile=False)
# Load the labels
class_names = open("model/Thumb_up_down/labels.txt", "r").readlines()
#0 Green, 1 Red
# class_names = ['Green', 'Red']

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, img = camera.read()
    h, w, _ = img.shape
    print(img.shape)  # 720 X 1280
    # print('img shape : ', img.shape[0])

    cx = h // 2
    cy = w // 2
    # img 720 X 1280
    # img = img[:, 100:100+img.shape[0]]
    img = img[:, 0:720]
    print('img', img.shape)

    # left_margin
    x1_left_margein = 170  # x : height
    y1_left_margin = 170  # y : width

    y1 = y1_left_margin
    y2 = img.shape[0] - y1_left_margin
    x1 = x1_left_margein
    x2 = img.shape[1] - x1_left_margein

    img = cv2.flip(img, 1)

    # Resize the raw image into (224-height,224-width) pixels
    input_img = cv2.resize(img, (224, 224))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Normalize the image array
    input_img = (input_img.astype(np.float32) / 127.5) - 1
    # print(input_img.shape)

    input_img = np.expand_dims(input_img, axis=0)
    # print(input_img.shape)

    # Predicts the model
    prediction = model.predict(input_img)
    print('prediction', prediction)

    index = np.argmax(prediction)
    idx = class_names[index]
    threshold = np.max(prediction)
    threshold = round(threshold, 2)
    print(idx, str(threshold)+'%')

    # rectangle
    rectangleColor = (0, 255, 0)
    rectangleThickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), rectangleColor, rectangleThickness)

    cv2.putText(img, idx, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(000, 000, 255), thickness=2)
    cv2.putText(img, text=str(round(threshold * 100, 2)) + "%", org=(170, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(000, 000, 255), thickness=2)
    # Show the image in a window
    cv2.imshow("Webcam Image", img)
    time.sleep(1 / 30)
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
