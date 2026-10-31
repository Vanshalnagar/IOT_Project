import re
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import easyocr


PROJECT_DIR = Path(__file__).resolve().parent
VEHICLE_MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
PLATE_MODEL_PATH = PROJECT_DIR / "license-plate-finetune-v1n.pt"
VEHICLE_CLASSES = {2, 3, 5, 7}
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SLOT_COUNT = 4
APP_CSS = """
<style>
:root { --ink:#e7eefb; --muted:#91a1bb; --panel:#111d31; --line:#24334c; --teal:#42d6bd; }
.stApp { background: radial-gradient(ellipse at 14% 0%, #142d3c 0, #0a1220 38%, #080e19 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stAppViewContainer"] .main .block-container { max-width:1420px; padding:2.2rem 3rem 4rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; color:#f2f6ff!important; }
h1 { font-size:2.4rem!important; }
p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#15243a,#101a2b); border:1px solid var(--line); padding:1rem 1.15rem; border-radius:16px; }
[data-testid="stMetricLabel"] { color:var(--muted)!important; }
[data-testid="stMetricValue"] { color:#f4f8ff!important; font-family:'Manrope',sans-serif; }
[data-testid="stCameraInput"] { background:#101a2b; border:1px solid var(--line); padding:1rem; border-radius:18px; }
[data-testid="stImage"] img { border-radius:16px; }
.hero { position:relative; overflow:hidden; padding:2rem 2.2rem; border:1px solid #264255; border-radius:24px; background:linear-gradient(115deg,rgba(17,44,59,.96),rgba(17,29,49,.92) 58%,rgba(24,43,59,.88)); margin-bottom:1.5rem; }
.hero:after { content:'P'; position:absolute; right:3rem; top:-5rem; font:800 18rem 'Manrope',sans-serif; color:rgba(66,214,189,.045); }
.eyebrow { color:var(--teal); font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { margin:.6rem 0 .35rem; font:800 2.25rem 'Manrope',sans-serif; color:#f3f7ff; letter-spacing:-.045em; }
.hero-copy { max-width:700px; margin:0; color:#aab8cc; font-size:1rem; }
.section-label { margin:1.6rem 0 .8rem; color:#e7eefb; font:700 1.15rem 'Manrope',sans-serif; }
.slot-card { min-height:122px; padding:1rem; border-radius:17px; border:1px solid var(--line); background:linear-gradient(145deg,#142138,#101a2a); }
.slot-card.available { border-color:#24534f; }
.slot-card.occupied { border-color:#51412d; }
.slot-top { display:flex; justify-content:space-between; align-items:center; }
.slot-name { font:800 1.25rem 'Manrope',sans-serif; color:#edf4ff; }
.slot-state { font-size:.72rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.slot-state.open { color:var(--teal); } .slot-state.busy { color:#ffbd75; }
.slot-plate { margin-top:.75rem; font-size:.9rem; color:#bfccde; overflow-wrap:anywhere; }
.subtle-card { padding:1rem 1.1rem; border:1px solid var(--line); border-radius:16px; background:rgba(17,29,49,.76); color:#aebbd0; }
.step { display:flex; gap:.7rem; margin:.6rem 0; color:#bfccde; font-size:.9rem; }
.step-number { color:var(--teal); font-weight:700; }
div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#24bca8,#49ddc3); border:0; color:#071520; font-weight:800; border-radius:12px; min-height:3rem; }
div.stButton > button[kind="primary"]:hover { background:#71ebd5; border:0; color:#071520; }
div.stButton > button:not([kind="primary"]) { border:1px solid #34445e; color:#cfdaea; border-radius:10px; }
[data-testid="stAlert"] { border-radius:14px; }
footer { visibility:hidden; }
</style>
"""


@st.cache_resource
def load_models():
    for model_path in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model file not found: {model_path}")

    return (
        YOLO(str(VEHICLE_MODEL_PATH)),
        YOLO(str(PLATE_MODEL_PATH)),
        easyocr.Reader(["en"], gpu=False),
    )


def scan_image(image_bytes, vehicle_model, plate_model, reader):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The captured image could not be decoded.")

    vehicle_result = vehicle_model(
        frame,
        imgsz=960,
        conf=0.15,
        iou=0.45,
        verbose=False,
    )[0]
    vehicles = [
        box for box in vehicle_result.boxes
        if int(box.cls[0]) in VEHICLE_CLASSES
    ]
    annotated = vehicle_result.plot()

    plate_result = plate_model(
        frame,
        imgsz=960,
        conf=0.12,
        iou=0.45,
        verbose=False,
    )[0]
    detected_plates = []

    for box in plate_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        enlarged = cv2.resize(
            crop,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)

        candidates = []
        for variant in (enlarged, enhanced):
            for item in reader.readtext(
                variant,
                allowlist=PLATE_ALLOWLIST,
                detail=1,
                paragraph=False,
            ):
                plate_text = re.sub(r"[^A-Z0-9]", "", item[1].upper())
                if len(plate_text) >= 5:
                    candidates.append((float(item[2]), plate_text))

        if candidates:
            _, plate_text = max(candidates)
            if plate_text not in detected_plates:
                detected_plates.append(plate_text)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                annotated,
                plate_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    return annotated, len(vehicles), detected_plates, len(plate_result.boxes)


def initialize_state():
    if "parking_slots" not in st.session_state:
        st.session_state.parking_slots = [None] * SLOT_COUNT
    else:
        st.session_state.parking_slots = [
            {"customer": "Existing booking", "plate": allocation}
            if isinstance(allocation, str)
            else allocation
            for allocation in st.session_state.parking_slots
        ]
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "registration_plate" not in st.session_state:
        st.session_state.registration_plate = ""


def render_slots():
    slots = st.session_state.parking_slots
    occupied = sum(allocation is not None for allocation in slots)
    available = len(slots) - occupied

    st.markdown(
        f'<div class="section-label">Parking lot <span style="color:#91a1bb;font:500 .9rem DM Sans">· {occupied} occupied / {available} available</span></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(SLOT_COUNT, gap="medium")
    for index, (column, allocation) in enumerate(zip(columns, slots), start=1):
        with column:
            is_available = allocation is None
            card_class = "available" if is_available else "occupied"
            status_class = "open" if is_available else "busy"
            status = "Available" if is_available else "Occupied"
            if is_available:
                plate_text = "Ready for assignment"
            else:
                plate_text = (
                    f"{escape(str(allocation['customer']))}<br>"
                    f"<span style='color:#8190a8'>"
                    f"{escape(str(allocation['plate']))}</span>"
                )
            st.markdown(
                f"""
                <div class="slot-card {card_class}">
                  <div class="slot-top">
                    <span class="slot-name">P{index:02d}</span>
                    <span class="slot-state {status_class}">● {status}</span>
                  </div>
                  <div class="slot-plate">{plate_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not is_available:
                if st.button("Release", key=f"release_slot_{index}"):
                    st.session_state.parking_slots[index - 1] = None
                    st.rerun()


def main():
    st.set_page_config(
        page_title="AI Smart Parking",
        page_icon="🚗",
        layout="wide",
    )
    initialize_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    occupied = sum(allocation is not None for allocation in st.session_state.parking_slots)

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Parking operations · AI assisted</div>
          <div class="hero-title">Smart Parking Console</div>
          <p class="hero-copy">Capture and inspect each arriving vehicle, then let an administrator assign the customer to an available parking bay.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("Total bays", SLOT_COUNT)
    metrics[1].metric("Occupied", occupied)
    metrics[2].metric("Available", SLOT_COUNT - occupied)

    try:
        vehicle_model, plate_model, reader = load_models()
    except Exception as exc:
        st.error(f"Could not load the AI models: {exc}")
        st.stop()

    camera_column, details_column = st.columns([1.1, 0.9], gap="large")
    with camera_column:
        st.markdown('<div class="section-label">Vehicle check-in</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtle-card"><div class="step"><span class="step-number">01</span><span>Allow browser camera access and frame the full vehicle.</span></div><div class="step"><span class="step-number">02</span><span>Keep its number plate close, sharp, and well lit.</span></div><div class="step"><span class="step-number">03</span><span>Review the AI result, then assign a bay manually.</span></div></div>',
            unsafe_allow_html=True,
        )
        captured_image = st.camera_input(
            "Capture vehicle image",
            key="parking_camera",
        )
        if captured_image is not None:
            if st.button("Analyze captured image", type="primary", width="stretch"):
                with st.spinner("Checking the vehicle and enhancing plate text..."):
                    try:
                        annotated, vehicle_count, plates, plate_region_count = scan_image(
                            captured_image.getvalue(),
                            vehicle_model,
                            plate_model,
                            reader,
                        )
                    except Exception as exc:
                        st.error(f"Image analysis failed: {exc}")
                    else:
                        plate = plates[0] if plates else ""
                        st.session_state.scan_result = {
                            "image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            "vehicle_count": vehicle_count,
                            "plate": plate,
                            "plate_region_count": plate_region_count,
                        }
                        st.session_state.registration_plate = plate
                        st.session_state.customer_name = ""

    with details_column:
        st.markdown('<div class="section-label">Detection & admin assignment</div>', unsafe_allow_html=True)
        result = st.session_state.scan_result
        if result is None:
            st.markdown(
                '<div class="subtle-card" style="min-height:180px;display:flex;flex-direction:column;justify-content:center;text-align:center"><div style="font-size:2.3rem">⌖</div><div style="font:700 1.05rem Manrope,sans-serif;color:#e7eefb;margin:.5rem">Ready for check-in</div><div>Capture and analyze a vehicle to begin.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.image(
                result["image"],
                caption="AI analysis overlay",
                width="stretch",
            )
            result_metrics = st.columns(2)
            result_metrics[0].metric("Vehicles", result["vehicle_count"])
            result_metrics[1].metric("Plate regions", result["plate_region_count"])
            if result["vehicle_count"] == 0 and result["plate_region_count"] == 0:
                st.warning(
                    "No vehicle or plate was detected. Use a closer, well-lit image "
                    "with the full vehicle and plate visible. You can still enter "
                    "the customer details, but verify the vehicle before assigning a bay."
                )
            elif result["vehicle_count"] == 0:
                st.warning("Plate region found, but vehicle detection missed. Verify the image before assigning.")
            elif not result["plate"]:
                st.warning("Vehicle found, but OCR could not read the plate. Enter it manually after checking the image.")
            else:
                st.success(f"Vehicle detected · OCR suggestion: {result['plate']}")

            available_slots = [
                index + 1
                for index, allocation in enumerate(st.session_state.parking_slots)
                if allocation is None
            ]
            if not available_slots:
                st.warning("All parking bays are occupied. Release a bay before assigning a customer.")
            else:
                with st.form("admin_assignment_form"):
                    st.markdown("**Manual customer assignment**")
                    st.text_input("Customer name", key="customer_name", placeholder="Enter customer name")
                    st.text_input(
                        "Vehicle registration",
                        key="registration_plate",
                        placeholder="Confirm OCR result or enter plate manually",
                    )
                    selected_slot = st.selectbox(
                        "Available parking bay",
                        options=available_slots,
                        format_func=lambda value: f"P{value:02d}",
                    )
                    verified = st.checkbox(
                        "I verified the customer, vehicle, and plate against the captured image."
                    )
                    submitted = st.form_submit_button(
                        "Assign customer to bay",
                        type="primary",
                        width="stretch",
                    )

                if submitted:
                    customer = st.session_state.customer_name.strip()
                    plate = re.sub(
                        r"[^A-Z0-9]",
                        "",
                        st.session_state.registration_plate.upper(),
                    )
                    if not customer:
                        st.error("Enter the customer name before assigning a bay.")
                    elif len(plate) < 5:
                        st.error("Enter a valid registration with at least five letters or digits.")
                    elif not verified:
                        st.error("Verify the customer and vehicle before assigning a parking bay.")
                    elif any(
                        allocation is not None and allocation["plate"] == plate
                        for allocation in st.session_state.parking_slots
                    ):
                        st.error(f"Registration {plate} is already assigned to a parking bay.")
                    elif st.session_state.parking_slots[selected_slot - 1] is not None:
                        st.error("That bay was just assigned. Choose another available bay.")
                    else:
                        st.session_state.parking_slots[selected_slot - 1] = {
                            "customer": customer,
                            "plate": plate,
                        }
                        st.success(f"{customer} assigned to bay P{selected_slot:02d}.")
                        st.rerun()

    st.info("Browser webcam capture only · No serial port or gate control")

    render_slots()
    st.markdown(
        '<p style="text-align:center;color:#72829b;font-size:.78rem;margin-top:2rem">SMART PARKING · LOCAL DEMO · Browser camera capture only</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
