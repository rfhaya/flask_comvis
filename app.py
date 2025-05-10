from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import threading
import time
from datetime import datetime
import os
import json
from PIL import Image
import io
import numpy as np
import base64
import tempfile
import uuid
import imageio.v2 as imageio

app = Flask(__name__)
model = YOLO("data/model/best.pt")

# === CCTV CONFIG ===
camera_configs = {
    "simpang_dharma3" : "https://cctv-stream.bandaacehkota.info/memfs/1e560ac1-8b57-416a-b64e-d4190ff83f88_output_0.m3u8",
    "simpang_dharma4" : "https://cctv-stream.bandaacehkota.info/memfs/f9444904-ad31-4401-9643-aee6e33b85c7_output_0.m3u8",
    "katamso_aniidrus" : "https://atcsdishub.pemkomedan.go.id/camera/KATAMSOANIIDRUS.m3u8",
    "katamso_masjidraya" : "https://atcsdishub.pemkomedan.go.id/camera/KATAMSOMASJIDRAYA.m3u8",
    "gelora1": "https://cctv.balitower.co.id/Gelora-017-700470_3/tracks-v1/index.fmp4.m3u8",
    "gelora2": "https://cctv.balitower.co.id/Gelora-017-700470_4/tracks-v1/index.fmp4.m3u8",
    "bendungan_hilir3": "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_2/tracks-v1/index.fmp4.m3u8"
}

camera_streams = {}
detections_per_camera = {}
DETECTIONS_FILE = "detections.json"
os.makedirs("static/detected", exist_ok=True)

# === JSON STORAGE ===
def save_detections_to_file():
    with open(DETECTIONS_FILE, "w") as f:
        json.dump(detections_per_camera, f)

def load_detections_from_file():
    global detections_per_camera
    if os.path.exists(DETECTIONS_FILE):
        with open(DETECTIONS_FILE, "r") as f:
            detections_per_camera = json.load(f)

# === IOU Logic ===
last_boxes = []
IOU_THRESHOLD = 0.5

def iou(b1, b2):
    xA = max(b1['x1'], b2['x1'])
    yA = max(b1['y1'], b2['y1'])
    xB = min(b1['x2'], b2['x2'])
    yB = min(b1['y2'], b2['y2'])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (b1['x2'] - b1['x1']) * (b1['y2'] - b1['y1'])
    boxBArea = (b2['x2'] - b2['x1']) * (b2['y2'] - b2['y1'])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union else 0

def is_duplicate(bbox):
    for prev in last_boxes:
        if iou(prev, bbox) > IOU_THRESHOLD:
            return True
    return False

def save_detection(label, conf, crop_img, bbox, camera_id, frame_count):
    folder = "static/detected"
    filename = f"{camera_id}_{int(time.time()*1000)}.jpg"
    abs_path = os.path.join(folder, filename)
    rel_path = f"detected/{filename}"

    saved = cv2.imwrite(abs_path, crop_img)
    print(f"💾 Saved: {saved} | Path: {abs_path} | Crop size: {crop_img.shape}")

    if not saved:
        return

    if camera_id not in detections_per_camera:
        detections_per_camera[camera_id] = []

    detections_per_camera[camera_id].append({
        "frame": frame_count,
        "class": label,
        "confidence": round(conf, 2),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "img": rel_path
    })

    detections_per_camera[camera_id] = detections_per_camera[camera_id][-100:]
    save_detections_to_file()

# === STREAM HANDLER ===
def stream_reader(camera_id, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ Failed to open stream {camera_id}")
        return
    while True:
        success, frame = cap.read()
        if not success:
            print(f"⚠️ No frame from {camera_id}")
            time.sleep(0.5)
            continue
        with camera_streams[camera_id]["lock"]:
            camera_streams[camera_id]["frame"] = frame.copy()
        time.sleep(0.03)

def initialize_streams():
    for cam_id, url in camera_configs.items():
        camera_streams[cam_id] = {"frame": None, "lock": threading.Lock()}
        t = threading.Thread(target=stream_reader, args=(cam_id, url), daemon=True)
        t.start()
        camera_streams[cam_id]["thread"] = t

def generate_frames(camera_id, apply_clahe=False, clip_limit=2.0, tile_size=8):
    frame_count = 0
    FRAME_SKIP = 2
    global last_boxes
    while True:
        with camera_streams[camera_id]["lock"]:
            frame = camera_streams[camera_id]["frame"]
            if frame is None:
                continue
            frame = frame.copy()

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        if apply_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        try:
            results = model.predict(frame, imgsz=640, conf=0.1, verbose=False, device=0)
            result = results[0]
        except Exception as e:
            print(f"❌ YOLO error [{camera_id}]:", e)
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_img = frame[y1:y2, x1:x2]
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if is_duplicate(bbox):
                continue
            save_detection(label, conf, crop_img, bbox, camera_id, frame_count)
            last_boxes.append(bbox)

        annotated = result.plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

# === VIDEO FEED ===
@app.route('/cctv/<camera_id>')
def camera_view(camera_id):
    data = detections_per_camera.get(camera_id, [])
    return render_template(f"cctv_{camera_id}.html", table_data=data)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id, apply_clahe=False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_clahe/<camera_id>')
def video_feed_clahe(camera_id):
    clip = float(request.args.get('clip', 2.0))
    tile = int(request.args.get('tile', 8))
    return Response(generate_frames(camera_id, apply_clahe=True, clip_limit=clip, tile_size=tile), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections/<camera_id>')
def get_detections(camera_id):
    return jsonify(detections_per_camera.get(camera_id, []))

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        if file:
            image_stream = file.stream.read()
            image = Image.open(io.BytesIO(image_stream)).convert('RGB')
            image_np = np.array(image)

            results = model.predict(image_np, imgsz=640, conf=0.25, verbose=False)
            annotated = results[0].plot()

            # 🔧 Konversi BGR ke RGB sebelum encode ke base64
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Encode hasil prediksi ke base64
            _, buffer = cv2.imencode('.jpg', annotated_rgb)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            return render_template(
                "input_image.html",
                original_b64=base64.b64encode(image_stream).decode('utf-8'),
                prediction_b64=img_b64
            )

        return "Invalid file", 400

    # Tampilkan form upload jika metode GET
    return render_template("input_image.html")

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'GET':
        return render_template("input_record.html")

    file = request.files.get('video')
    if not file or file.filename == '':
        return render_template("input_record.html", error="No video uploaded")

    clip_limit = float(request.form.get('clip', 2.0))
    tile_size = int(request.form.get('tile', 8))

    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    # Simpan video asli
    original_path = os.path.join("static/uploads", file.filename)
    file.save(original_path)

    cap = cv2.VideoCapture(original_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or np.isnan(fps):
        fps = 25

    output_filename = f"static/results/result_{int(time.time())}.mp4"

    # 👇 Buat writer dengan H.264 codec
    writer = imageio.get_writer(output_filename, fps=fps, codec='libx264')

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # YOLOv8
        results = model.predict(enhanced, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()

        # 👇 Simpan ke video (RGB harus dipakai di imageio)
        writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    cap.release()
    writer.close()

    return render_template("input_record.html",
                           original_video=original_path.split("static/")[1],
                           predicted_video=output_filename.split("static/")[1])
# === MAIN ===
if __name__ == '__main__':
    initialize_streams()
    app.run(debug=True)
