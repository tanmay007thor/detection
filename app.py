from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    PTime = 0

    while True:
        success, img = cap.read()
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRgb)

        if results.multi_hand_landmarks:
            for HandLms in results.multi_hand_landmarks:
                for id, lm in enumerate(HandLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, HandLms, mpHands.HAND_CONNECTIONS)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
