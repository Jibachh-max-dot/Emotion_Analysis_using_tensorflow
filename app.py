import cv2
import numpy as np 
import tensorflow
import google.protobuf
import streamlit as st 
from keras.models import model_from_json

#load the model 

with open('models/model.json','r') as model_json_file:
    model_json = model_json_file.read()
    model = model_from_json(model_json)
    model.load_weights("models/model.h5")
    
#define the emotions
emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']

#create a funtions to predict the emotions

def predict_emotion(img):
    img = cv2.resize(img, (48,48))
    img = np.reshape(img, [1, 48, 48, 1])
    prediction = model.predict(img)
    return emotions[np.argmax(prediction)]

#create a funtion to detect the face in the frame

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return gray[y:y+h, x:x+w], faces[0]
    else:
        return None, None

#create the streamlit app

st.set_page_config(page_title="Emotion Detection", page_icon=":guardsman:", layout="wide")
st.title("Emotion Detection using OpenCV and Keras")

#add a checkbox to toggle  the webcam

webcam_enabled = st.checkbox("Enable Webcam")

if webcam_enabled:
    webcam = cv2.VideoCapture(0)
    while True:
        _, frame = webcam.read()
        face, rect = detect_face(frame)
        if face is not None:
            emotion = predict_emotion(face)
            (x, y, w, h) = rect
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0,255,0],2)
            cv2.rectangle(frame,(x, y), (x+w, y+h),(0, 0, 255),2)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
else:
    cv2.destroyAllWindows()
