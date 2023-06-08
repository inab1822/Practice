from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Load the model
model = load_model("model/Thumb_up_down/keras_Model.h5", compile=False)

# Load the labels
# class_names = open("labels.txt", "r").readlines()
class_name = ['Thump-up', 'Thumb-down', 'Background']
print(class_name)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # flip image
    image = cv2.flip(image, 1)

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224*2, 224*2), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    img_input =  img_input = cv2.resize(image, (224, 224))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    # Normalize the image array
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    img_input = np.expand_dims(img_input, axis=0)

    # Predicts the model
    prediction = model.predict(img_input)
    idx = np.argmax(prediction)
    confidence_score = prediction[0][idx]

    cv2.putText(image, text=class_name[idx],
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2)
    # Show the image in a window
    cv2.imshow("Webcam Image", image)


    # Print prediction and confidence score
    print("Class:", class_name[idx])
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
