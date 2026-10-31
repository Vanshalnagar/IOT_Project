
import cv2

CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open external webcam on Camera 1")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("External USB Webcam Connected!")
print("Press Q to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame")
        break

    cv2.putText(
        frame,
        "EXTERNAL USB CAMERA - INDEX 1",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("External USB Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera Closed!")
