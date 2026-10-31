"""
AI Automatic Parking Gate - Python application

Expected files in this same folder:
  yolo11n.pt
  license-plate-finetune-v1n.pt

Install packages in your activated virtual environment:
  python -m pip install --index-url https://pypi.org/simple opencv-python ultralytics pyserial easyocr

Start this Python program AFTER uploading the ESP32 firmware.
Close Arduino IDE Serial Monitor/Plotter before running this program.
"""

import re
import time
from pathlib import Path

import cv2
import easyocr
import serial
import serial.tools.list_ports
from ultralytics import YOLO


# -------------------- USER SETTINGS --------------------

PROJECT_DIR = Path(__file__).resolve().parent

VEHICLE_MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
PLATE_MODEL_PATH = PROJECT_DIR / "license-plate-finetune-v1n.pt"

SERIAL_PORT = "COM6"       # Change this if Device Manager shows another port.
BAUD_RATE = 115200
CAMERA_INDEX = 0           # Try 1 if your external USB webcam is camera 1.

SCAN_SECONDS = 15
VEHICLE_CONFIDENCE = 0.40
PLATE_CONFIDENCE = 0.25

# Basic Indian registration format check. This is NOT an ownership check.
PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# Optional security whitelist:
# Example: ALLOWED_PLATES = {"GJ01AB1234", "MH12CD5678"}
# Leave empty to accept any plate that matches the format above.
ALLOWED_PLATES = set()

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}


# -------------------- SERIAL CONNECTION --------------------

def list_serial_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Check the ESP32 USB cable and driver.")
        return

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")


def connect_esp32():
    list_serial_ports()

    try:
        esp = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.2,
            write_timeout=1
        )
        # Opening a serial connection may reset some ESP32 boards.
        time.sleep(2.5)
        esp.reset_input_buffer()
        print(f"ESP32 connected on {SERIAL_PORT}")
        return esp

    except serial.SerialException as exc:
        raise RuntimeError(
            f"Could not open {SERIAL_PORT}: {exc}\n"
            "Close Arduino Serial Monitor/Plotter and any other program "
            "using this COM port. Confirm the port in Device Manager."
        ) from exc


def send_open_command(esp):
    esp.write(b"OPEN\n")
    esp.flush()
    print("OPEN command sent to ESP32.")


# -------------------- LICENSE PLATE HELPERS --------------------

def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def plate_is_accepted(plate):
    if not PLATE_PATTERN.fullmatch(plate):
        return False

    if ALLOWED_PLATES and plate not in ALLOWED_PLATES:
        print(f"Plate {plate} is not in the authorized list.")
        return False

    return True


# -------------------- CAMERA / DETECTION --------------------

def scan_vehicle(vehicle_model, plate_model, reader):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        print(
            f"Webcam {CAMERA_INDEX} could not be opened. "
            "Try CAMERA_INDEX = 0 or 1."
        )
        return None

    print("Webcam activated. Scanning for a vehicle and number plate...")
    start_time = time.time()
    accepted_plate = None

    try:
        while time.time() - start_time < SCAN_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the webcam.")
                break

            vehicle_result = vehicle_model(frame, verbose=False)[0]

            vehicle_boxes = [
                box for box in vehicle_result.boxes
                if int(box.cls[0]) in VEHICLE_CLASSES
                and float(box.conf[0]) >= VEHICLE_CONFIDENCE
            ]

            annotated = vehicle_result.plot()
            vehicle_count = len(vehicle_boxes)
            best_candidate = None

            if vehicle_count > 0:
                plate_result = plate_model(
                    frame,
                    conf=PLATE_CONFIDENCE,
                    verbose=False
                )[0]

                for box in plate_result.boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    height, width = frame.shape[:2]
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    # Upscale small plate crops to help OCR.
                    plate_crop = cv2.resize(
                        plate_crop,
                        None,
                        fx=2,
                        fy=2,
                        interpolation=cv2.INTER_CUBIC
                    )

                    ocr_results = reader.readtext(
                        plate_crop,
                        detail=1,
                        paragraph=False
                    )

                    for ocr_item in ocr_results:
                        raw_text = ocr_item[1]
                        cleaned = normalize_plate(raw_text)

                        if plate_is_accepted(cleaned):
                            best_candidate = cleaned
                            break

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    if best_candidate:
                        cv2.putText(
                            annotated,
                            best_candidate,
                            (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2
                        )
                        break

            if best_candidate:
                accepted_plate = best_candidate

            cv2.putText(
                annotated,
                f"Vehicles: {vehicle_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            status = accepted_plate if accepted_plate else "Scanning..."
            cv2.putText(
                annotated,
                f"Plate: {status}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.imshow("AI Parking - Vehicle Detection", annotated)

            if accepted_plate:
                print(f"Recognized number plate: {accepted_plate}")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Scanning cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return accepted_plate


# -------------------- MAIN APPLICATION --------------------

def main():
    for model_path in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Put both .pt model files in the same folder as detection.py."
            )

    print("Loading YOLO vehicle model...")
    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))

    print("Loading license plate model...")
    plate_model = YOLO(str(PLATE_MODEL_PATH))

    print("Loading EasyOCR (first run may download model files)...")
    reader = easyocr.Reader(["en"], gpu=False)

    print("All AI models loaded.")
    esp = connect_esp32()

    print("\n======================================")
    print("   AI AUTOMATIC PARKING GATE SYSTEM")
    print("======================================")
    print("Waiting for ESP32 ultrasonic detection...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            message = esp.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if not message:
                continue

            # Uncomment to see all ESP32 messages:
            # print("ESP32:", message)

            if message == "DETECTED":
                print("\nVehicle detected by ultrasonic sensor.")
                plate = scan_vehicle(
                    vehicle_model, plate_model, reader
                )

                if plate and plate_is_accepted(plate):
                    print("--------------------------------------")
                    print("VEHICLE PLATE:", plate)
                    print("--------------------------------------")
                    send_open_command(esp)
                    print("ESP32 will close the gate after its open timer.")
                else:
                    print("No acceptable plate found. Gate remains closed.")

                print("Waiting for the next vehicle...\n")

    finally:
        if esp and esp.is_open:
            esp.close()
        cv2.destroyAllWindows()
        print("Serial connection closed. Program stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem stopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
