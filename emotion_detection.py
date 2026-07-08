import cv2
from fer import FER
emotion_detector = FER()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    emotions = emotion_detector.detect_emotions(frame)

    for emotion in emotions:
        x, y, w, h = emotion["box"]

        emotion_name = max(emotion["emotions"], key=emotion["emotions"].get)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, emotion_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("AI Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()