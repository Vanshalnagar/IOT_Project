
import cv2
import re
import easyocr
from ultralytics import YOLO

# Load YOLO models
vehicle_model = YOLO("yolo11n.pt")
plate_model = YOLO("license-plate-finetune-v1n.pt")

# Load OCR
reader = easyocr.Reader(['en'], gpu=False)

# Open external USB webcam
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open external webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

vehicle_classes = [2, 3, 5, 7]

print("Camera Started!")
print("Car and Number Plate Detection Started!")
print("Press Q to exit.")

last_plate = ""

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error!")
        break

    display = frame.copy()

    # =================================
    # 1. DETECT VEHICLES
    # =================================

    vehicle_results = vehicle_model(
        frame,
        conf=0.35,
        verbose=False
    )[0]

    vehicle_count = 0

    for box in vehicle_results.boxes:

        class_id = int(box.cls[0])

        if class_id not in vehicle_classes:
            continue

        vehicle_count += 1

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            display,
            "VEHICLE",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # =================================
    # 2. DETECT NUMBER PLATES
    # =================================

    plate_results = plate_model(
        frame,
        conf=0.25,
        verbose=False
    )[0]

    for box in plate_results.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Enlarge plate for better OCR
        plate_crop = cv2.resize(
            plate_crop,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # =================================
        # 3. READ NUMBER PLATE USING OCR
        # =================================

        ocr_results = reader.readtext(
            plate_crop,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for result in ocr_results:

            raw_text = result[1]

            plate_number = re.sub(
                r'[^A-Z0-9]',
                '',
                raw_text.upper()
            )

            if len(plate_number) >= 5:

                last_plate = plate_number

                print("NUMBER PLATE:", plate_number)

                # Draw plate rectangle
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    display,
                    plate_number,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    # =================================
    # 4. DISPLAY INFORMATION
    # =================================

    cv2.putText(
        display,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    if last_plate:

        cv2.putText(
            display,
            f"NUMBER PLATE: {last_plate}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "CAR AND NUMBER PLATE RECOGNITION",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera Closed!")
