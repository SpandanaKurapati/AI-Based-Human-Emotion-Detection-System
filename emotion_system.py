import cv2
from fer import FER

detector = FER()

print("Emotion Detection System")
print("1 - Detect Emotion from Camera")
print("2 - Detect Emotion from Image")
print("3 - Detect Emotion from Video")

choice = input("Enter your choice (1/2/3): ")

# ---------------- CAMERA ----------------
if choice == "1":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = detector.detect_emotions(frame)

        if emotions:
            x, y, w, h = emotions[0]["box"]
            emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Camera Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- IMAGE ----------------
elif choice == "2":
    image_path = input("Enter image file path: ")
    img = cv2.imread(image_path)

    emotions = detector.detect_emotions(img)

    if emotions:
        x, y, w, h = emotions[0]["box"]
        emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Image Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- VIDEO ----------------
elif choice == "3":
    video_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = detector.detect_emotions(frame)

        if emotions:
            x, y, w, h = emotions[0]["box"]
            emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Video Emotion Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice!")