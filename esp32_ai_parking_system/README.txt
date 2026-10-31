AI Automatic Parking Gate - setup

FILES
- parking_gate_esp32.ino: upload this to your ESP32 using Arduino IDE.
- detection.py: run this on Windows after uploading the ESP32 firmware.

MODEL FILES
Put these two files in the same folder as detection.py:
- yolo11n.pt
- license-plate-finetune-v1n.pt

PYTHON SETUP (Windows CMD, from the project folder)
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple opencv-python ultralytics pyserial easyocr

HARDWARE ASSUMPTIONS
For a classic ESP32 DevKit / ESP32-WROOM-32:
- HC-SR04 TRIG -> GPIO 5
- HC-SR04 ECHO -> GPIO 19 through a voltage divider (do not feed 5V into ESP32 GPIO)
- Servo signal -> GPIO 18
- Servo gets a separate suitable 5V supply.
- Join ESP32 GND, sensor GND and servo supply GND.
Check your board pinout before wiring; ESP32 variants differ.

ARDUINO SETUP
1. Install the ESP32 board package in Arduino IDE.
2. Install ESP32Servo using Library Manager.
3. Select your actual ESP32 board and COM port.
4. Upload parking_gate_esp32.ino.
5. Close Arduino Serial Monitor and Serial Plotter.

RUN
1. Connect the ESP32 and leave it powered.
2. Activate venv.
3. Run: python detection.py
4. If needed, edit SERIAL_PORT and CAMERA_INDEX at the top of detection.py.
5. Start the vehicle only after Python says it is waiting for ultrasonic detection.

SERIAL PROTOCOL
ESP32 sends: DETECTED
Python sends: OPEN
ESP32 closes the gate after 3 seconds.

SAFETY / LIMITATIONS
- The code opens the gate for any recognized plate matching a basic Indian registration-number pattern by default.
- This format check does not verify ownership or authorization. For real access control, populate ALLOWED_PLATES in detection.py with authorized registrations.
- Test with the servo arm disconnected or gate mechanism unloaded first. Add physical obstruction detection / safety sensors before using a real gate.
- Do not power a servo from the ESP32 3.3V pin. Use a suitable external supply and common ground.
- HC-SR04 ECHO is commonly 5V; use a voltage divider or level shifter before the ESP32 input.
- Pin assignments and servo angles may need adjustment for your hardware.
