import cv2
import torch
import numpy as np
import threading
import time
import pyttsx3
import os
import logging
import queue
import json
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, 
                   async_mode='threading',
                   cors_allowed_origins="*",
                   logger=True,
                   engineio_logger=True)

# ========== SIMPLIFIED CAMERA HANDLING ==========
class SimpleCamera:
    def __init__(self):
        self.cap = None
        self.current_source = 0
        self.lock = threading.Lock()
        self.last_frame_time = time.time()

    def open_camera(self, source):
        with self.lock:
            # Release existing camera
            if self.cap and self.cap.isOpened():
                self.cap.release()
                
            # Try to open new camera
            try:
                if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                    src = int(source)
                    self.cap = cv2.VideoCapture(src)
                elif source.startswith(("http", "rtsp")):
                    self.cap = cv2.VideoCapture(source)
                else:
                    logging.error(f"Unsupported camera source: {source}")
                    return False
                
                # Set camera properties
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 15)
                    self.current_source = source
                    logging.info(f"Camera opened: {source}")
                    return True
                else:
                    logging.error(f"Failed to open camera: {source}")
                    return False
            except Exception as e:
                logging.error(f"Camera open error: {str(e)}")
                return False
    
    def read_frame(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                try:
                    success, frame = self.cap.read()
                    if success:
                        self.last_frame_time = time.time()
                        return frame
                    else:
                        # Coba buka ulang kamera jika gagal membaca frame
                        self.open_camera(self.current_source)
                        return None
                except Exception as e:
                    logging.error(f"Frame read error: {str(e)}")
                    return None
            return None
    
    def is_active(self):
        return (time.time() - self.last_frame_time) < 5
    
    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None

# Initialize camera
camera = SimpleCamera()
camera.open_camera(0)  # Open default camera

# ========== ENHANCED CONFIGURATION ==========
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Model configuration
MODEL_CONFIG = {
    'yolov8n': {
        'path': os.path.join('models', 'yolov8n.pt'),
        'name': 'YOLOv8 Nano',
        'speed': '45-60 FPS',
        'info': 'Kecepatan tinggi, akurasi baik'
    },
    'ssd_mobilenet': {
        'path': os.path.join('models', 'ssd_mobilenet.onnx'),
        'name': 'MobileNet-SSD',
        'speed': '60-80 FPS',
        'info': 'Tercepat untuk perangkat lemah'
    },
    'yolov5n': {
        'path': os.path.join('models', 'yolov5n.pt'),
        'name': 'YOLOv5 Nano',
        'speed': '30-40 FPS',
        'info': 'Kompatibilitas luas'
    }
}

# COCO class names for MobileNet-SSD
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Current model state
current_model_name = 'yolov8n'
current_model = None
model_classes = {}

# Load model function
def load_model(model_name):
    global current_model, model_classes
    
    logging.info(f"Loading model: {model_name}")
    
    if model_name == 'yolov8n':
        # Load YOLOv8 Nano
        model = YOLO(MODEL_CONFIG[model_name]['path'])
        model.fuse()
        model_classes = model.names
        logging.info(f"YOLOv8n model loaded with {len(model_classes)} classes")
        
    elif model_name == 'ssd_mobilenet':
        # Load MobileNet-SSD
        model = cv2.dnn.readNetFromONNX(MODEL_CONFIG[model_name]['path'])
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        model_classes = COCO_CLASSES
        logging.info(f"MobileNet-SSD model loaded with {len(model_classes)} classes")
        
    elif model_name == 'yolov5n':
        # Load YOLOv5 Nano
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=MODEL_CONFIG[model_name]['path'], 
                              force_reload=True)
        model.to(device)
        model.conf = 0.6
        model_classes = model.names
        logging.info(f"YOLOv5n model loaded with {len(model_classes)} classes")
    
    else:
        logging.error(f"Unknown model: {model_name}")
        return None
    
    return model

# Initialize the default model
current_model = load_model(current_model_name)

# Enhanced TTS with queue system
tts_queue = queue.Queue()

def tts_worker():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    logging.info("TTS worker aktif")

    while True:
        message = tts_queue.get()
        if message is None:
            break
        try:
            logging.debug(f"[TTS] Membaca: {message}")
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {str(e)}")
        tts_queue.task_done()


# Start TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Distance estimation configuration
KNOWN_WIDTHS_JSON = 'known_widths.json'
KNOWN_WIDTHS = {}

# Load known widths from JSON
try:
    if os.path.exists(KNOWN_WIDTHS_JSON):
        with open(KNOWN_WIDTHS_JSON, 'r') as f:
            KNOWN_WIDTHS = json.load(f)
        logging.info(f"Loaded {len(KNOWN_WIDTHS)} known widths from {KNOWN_WIDTHS_JSON}")
    else:
        # Default values if JSON not found
        KNOWN_WIDTHS = {
            'cell phone': 15,
            'cup': 8,
            'bottle': 7,
            'person': 50,
            'chair': 40,
            'book': 20,
            'keyboard': 40,
            'laptop': 35,
            'tv': 80
        }
        logging.warning(f"Known widths file not found, using defaults")
except Exception as e:
    logging.error(f"Error loading known widths: {str(e)}")
    KNOWN_WIDTHS = {}

focal_length = None

# Calibration constant
KNOWN_DISTANCE = 100  # Real-world distance in cm for calibration

# Global variables
camera_source = 0
detection_active = True
last_announce_time = 0
cooldown = 5          # Cooldown for announcements
last_main_object = None  # Track last announced object
frame_counter = 0

def estimate_distance(label, perceived_width):
    """Estimate distance based on perceived width in frame"""
    if perceived_width <= 0 or focal_length is None:
        return None
    
    # Find known width for this label
    known_width = None
    
    # First try exact match
    if label.lower() in KNOWN_WIDTHS:
        known_width = KNOWN_WIDTHS[label.lower()]
    else:
        # Try partial match
        for key in KNOWN_WIDTHS:
            if key in label.lower() or label.lower() in key:
                known_width = KNOWN_WIDTHS[key]
                break
    
    if not known_width:
        return None
    
    distance = (known_width * focal_length) / perceived_width
    return float(distance)  # Convert to Python float

def calibrate_focal_length(measured_distance, real_width, width_in_frame):
    """Calibrate focal length using reference object"""
    return (width_in_frame * measured_distance) / real_width

def process_detections(frame, model_name):
    """Process detections based on the active model"""
    global focal_length
    
    detected_objects = []
    h, w = frame.shape[:2]
    
    if model_name == 'yolov8n':
        # YOLOv8 detection
        results = current_model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_idx = det[:6]
            label = model_classes[int(cls_idx)]
            width = x2 - x1
            
            # Calibrate on first cell phone detection
            if label in KNOWN_WIDTHS and width > 10 and focal_length is None:
                focal_length = calibrate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTHS[label], width)
                logging.info(f"Focal length calibrated from {label}: {focal_length:.2f}")
            
            # Estimate distance
            distance = estimate_distance(label, width)
            
            detected_objects.append({
                'label': label,
                'confidence': float(conf),  # Convert to Python float
                'distance': distance,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'width': float(width)  # Convert to Python float
            })
                    
            # Draw bounding boxes and labels
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label_text = f"{label} {conf:.2f}"
            if distance:
                label_text += f" | {distance:.1f}cm"
            cv2.putText(frame, label_text, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Trigger TTS announcements
        announce_detections(detected_objects)    
    elif model_name == 'ssd_mobilenet':
        # MobileNet-SSD detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        current_model.setInput(blob)
        detections = current_model.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id < len(model_classes):
                    label = model_classes[class_id]
                    
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype('int')
                    width = x2 - x1
                    
                    # Calibrate on first cell phone detection
                    if label in KNOWN_WIDTHS and width > 10 and focal_length is None:
                        focal_length = calibrate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTHS[label], width)
                        logging.info(f"Focal length calibrated from {label}: {focal_length:.2f}")
                    
                    # Estimate distance
                    distance = estimate_distance(label, width)
                    
                    detected_objects.append({
                        'label': label,
                        'confidence': float(confidence),  # Convert to Python float
                        'distance': distance,
                        'bbox': [x1, y1, x2, y2],
                        'width': float(width)  # Convert to Python float
                    })

                    
                    # Draw bounding boxes and labels
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} {confidence:.2f}"
                    if distance:
                        label_text += f" | {distance:.1f}cm"
                    cv2.putText(frame, label_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Trigger TTS announcements
        announce_detections(detected_objects)    
    elif model_name == 'yolov5n':
        # YOLOv5 detection
        results = current_model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        for *xyxy, conf, cls in detections:
            label = model_classes[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)
            width = x2 - x1
            
            # Calibrate on first cell phone detection
            if label in KNOWN_WIDTHS and width > 10 and focal_length is None:
                focal_length = calibrate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTHS[label], width)
                logging.info(f"Focal length calibrated from {label}: {focal_length:.2f}")
            
            # Estimate distance
            distance = estimate_distance(label, width)
            
            detected_objects.append({
                'label': label,
                'confidence': float(conf),  # Convert to Python float
                'distance': distance,
                'bbox': [x1, y1, x2, y2],
                'width': float(width)  # Convert to Python float
            })
            # Draw bounding boxes and labels
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {conf:.2f}"
            if distance:
                label_text += f" | {distance:.1f}cm"
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Trigger TTS announcements
        announce_detections(detected_objects)    
    return detected_objects

def generate_frames():
    global focal_length, last_main_object
    frame_count = 0
    
    logging.info("Video streaming started")
    
    while True:
        frame = camera.read_frame()
        
        if frame is None:
            # Create more informative blank frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Tampilkan status kamera
            if camera.cap is None:
                status_text = "Kamera tidak terhubung"
            elif not camera.cap.isOpened():
                status_text = "Gagal membuka kamera"
            else:
                status_text = "Menerima sinyal kamera..."
                
            cv2.putText(frame, status_text, (50, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Tambahkan petunjuk
            cv2.putText(frame, "Periksa koneksi kamera", (50, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame_count += 1
            
            # Skip processing every 2nd frame for performance
            process_frame = detection_active and (frame_count % 2 == 0)
            
            detected_objects = []
            if process_frame:
                try:
                    detected_objects = process_detections(frame, current_model_name)
                    
                    # Update frontend with detections
                    socketio.emit('detection_update', {'objects': detected_objects})
                    
                except Exception as e:
                    logging.error(f"Detection error: {str(e)}")
        
        # Encode and stream frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
def announce_detections(current_objects):
    """Announce detections with cooldown and change detection"""
    global last_announce_time, last_main_object

    if not current_objects:
        last_main_object = None
        return

    current_time = time.time()

    # Optional: Reset last_main_object after 15 seconds of silence
    if (current_time - last_announce_time) > 5:
        logging.debug("Auto-reset last_main_object after inactivity")
        last_main_object = None

    # Get the object with highest confidence
    main_obj = max(current_objects, key=lambda x: x['confidence'])

    object_changed = True
    if last_main_object:
        same_label = main_obj['label'] == last_main_object['label']
        similar_distance = True
        similar_size = True

        if main_obj.get('distance') and last_main_object.get('distance'):
            dist_diff = abs(main_obj['distance'] - last_main_object['distance'])
            avg_dist = (main_obj['distance'] + last_main_object['distance']) / 2
            similar_distance = (dist_diff / avg_dist) < 0.3 if avg_dist > 0 else True

        if main_obj.get('width') and last_main_object.get('width'):
            size_diff = abs(main_obj['width'] - last_main_object['width'])
            avg_size = (main_obj['width'] + last_main_object['width']) / 2
            similar_size = (size_diff / avg_size) < 0.3 if avg_size > 0 else True

        object_changed = not (same_label and similar_distance and similar_size)

        logging.debug(f"Label sama: {same_label}")
        logging.debug(f"Ukuran mirip: {similar_size}")
        logging.debug(f"Jarak mirip: {similar_distance}")

    cooldown_passed = (current_time - last_announce_time) > cooldown

    logging.debug(f"Main object: {main_obj}")
    logging.debug(f"Last object: {last_main_object}")
    logging.debug(f"Cooldown passed: {cooldown_passed}")
    logging.debug(f"Object changed: {object_changed}")

    # 👉 Jika objek baru (label berubah), langsung umumkan meskipun cooldown belum selesai
    if object_changed and (cooldown_passed or not last_main_object or main_obj['label'] != last_main_object['label']):
        message = f"Ada {main_obj['label']} di depan Anda"
        if main_obj.get('distance'):
            message += f" sekitar {int(main_obj['distance'])} sentimeter"

        tts_queue.put(message)
        logging.info(f"Announcing: {message}")

        last_announce_time = current_time
        last_main_object = main_obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global detection_active
    
    data = request.json
    new_source = data.get('camera_source', camera.current_source)
    detection_active = data.get('detection_active', detection_active)
    
    # Update camera source
    if new_source != camera.current_source:
        success = camera.open_camera(new_source)
        if not success:
            socketio.emit('camera_error', {'message': f'Failed to open camera: {new_source}'})
    
    logging.info(f"Settings updated: camera={camera.current_source}, detection_active={detection_active}")
    return jsonify(success=True, camera_source=camera.current_source)

@app.route('/set_model', methods=['POST'])
def set_model():
    """Endpoint to change the active model"""
    global current_model, current_model_name
    
    model_name = request.json.get('model_name')
    if model_name not in MODEL_CONFIG:
        return jsonify(success=False, error="Invalid model name"), 400
    
    try:
        new_model = load_model(model_name)
        if new_model:
            current_model = new_model
            current_model_name = model_name
            logging.info(f"Model switched to: {model_name}")
            return jsonify(
                success=True, 
                model_name=model_name,
                model_info=MODEL_CONFIG[model_name]
            )
        return jsonify(success=False, error="Failed to load model"), 500
    except Exception as e:
        logging.error(f"Model switch error: {str(e)}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/trigger_tts', methods=['POST'])
def trigger_tts():
    """Manual TTS trigger endpoint"""
    data = request.json
    message = data.get('message', '')
    
    if message:
        tts_queue.put(message)
        return jsonify(success=True)
    return jsonify(success=False, error="Empty message"), 400

@app.route('/update_known_width', methods=['POST'])
def update_known_width():
    """Update or add a known width entry"""
    data = request.json
    label = data.get('label')
    width = data.get('width')
    
    if not label or not width:
        return jsonify(success=False, error="Missing label or width"), 400
    
    try:
        width = float(width)
    except ValueError:
        return jsonify(success=False, error="Invalid width value"), 400
    
    # Update in-memory dictionary
    KNOWN_WIDTHS[label.lower()] = width
    
    # Update JSON file
    try:
        with open(KNOWN_WIDTHS_JSON, 'w') as f:
            json.dump(KNOWN_WIDTHS, f, indent=2)
        return jsonify(success=True)
    except Exception as e:
        logging.error(f"Failed to update known widths: {str(e)}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/get_known_widths')
def get_known_widths():
    return jsonify(KNOWN_WIDTHS)

@app.route('/get_current_model')
def get_current_model():
    return jsonify({
        'model_name': current_model_name,
        'model_info': MODEL_CONFIG[current_model_name]
    })

@app.route('/camera_status')
def camera_status():
    try:
        # Coba baca frame sebagai tes
        frame = camera.read_frame()
        return jsonify({
            'active': frame is not None,
            'source': camera.current_source
        })
    except Exception as e:
        logging.error(f"Camera status check error: {str(e)}")
        return jsonify({
            'active': False,
            'error': str(e)
        }), 500  # Tambahkan status code 500 untuk error
    
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Cleanup on exit
def cleanup():
    tts_queue.put(None)  # Signal worker to exit
    tts_thread.join(timeout=1.0)
    camera.release()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=5000, 
                 debug=True, 
                 allow_unsafe_werkzeug=True,
                 use_reloader=False)