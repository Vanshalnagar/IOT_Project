/*
  AI Automatic Parking Gate - ESP32 firmware

  Wiring (classic ESP32 DevKit / ESP32-WROOM-32):
    HC-SR04 TRIG -> GPIO 5
    HC-SR04 ECHO -> GPIO 19 (use a voltage divider: HC-SR04 ECHO is 5V)
    Servo signal -> GPIO 18
    Servo power -> external regulated 5V supply
    ESP32 GND, sensor GND and servo supply GND must be connected together.

  USB serial protocol at 115200 baud:
    ESP32 -> PC: DETECTED
    PC -> ESP32: OPEN

  Install the "ESP32Servo" library in Arduino IDE Library Manager.
*/

#include <Arduino.h>
#include <ESP32Servo.h>

const int TRIG_PIN = 5;
const int ECHO_PIN = 19;
const int SERVO_PIN = 18;

const int GATE_CLOSED_ANGLE = 0;
const int GATE_OPEN_ANGLE = 90;

const float DETECT_DISTANCE_CM = 25.0;
const float CLEAR_DISTANCE_CM = 40.0;
const unsigned long OPEN_DURATION_MS = 3000;
const unsigned long SENSOR_INTERVAL_MS = 100;
const unsigned long SERIAL_TIMEOUT_MS = 30;

Servo gateServo;

bool vehicleLatched = false;
bool gateIsOpen = false;
unsigned long gateOpenedAt = 0;
unsigned long lastSensorReadAt = 0;

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Timeout prevents the ESP32 from blocking if no echo is received.
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) {
    return -1.0;
  }

  return (duration * 0.0343f) / 2.0f;
}

void openGate() {
  if (gateIsOpen) {
    return;
  }

  gateServo.write(GATE_OPEN_ANGLE);
  gateIsOpen = true;
  gateOpenedAt = millis();
  Serial.println("GATE_OPENED");
}

void closeGate() {
  gateServo.write(GATE_CLOSED_ANGLE);
  gateIsOpen = false;
  Serial.println("GATE_CLOSED");
}

void handleSerialCommands() {
  static String command = "";

  while (Serial.available() > 0) {
    char ch = (char)Serial.read();

    if (ch == '\n' || ch == '\r') {
      command.trim();

      if (command.equalsIgnoreCase("OPEN")) {
        openGate();
      }

      command = "";
    } else if (command.length() < 40) {
      command += ch;
    } else {
      command = "";
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  gateServo.setPeriodHertz(50);
  gateServo.attach(SERVO_PIN, 500, 2400);
  gateServo.write(GATE_CLOSED_ANGLE);

  delay(500);
  Serial.println("PARKING_GATE_READY");
}

void loop() {
  handleSerialCommands();

  // Automatically close the gate after the open interval.
  if (gateIsOpen && millis() - gateOpenedAt >= OPEN_DURATION_MS) {
    closeGate();
  }

  if (millis() - lastSensorReadAt >= SENSOR_INTERVAL_MS) {
    lastSensorReadAt = millis();

    float distance = readDistanceCm();

    if (distance > 0 && distance <= DETECT_DISTANCE_CM && !vehicleLatched) {
      vehicleLatched = true;
      Serial.println("DETECTED");
    }

    // Require the vehicle to move farther away before another detection.
    if (distance < 0 || distance >= CLEAR_DISTANCE_CM) {
      vehicleLatched = false;
    }
  }
}
