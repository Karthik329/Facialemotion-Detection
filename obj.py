import cv2
import numpy as np
import time
import math
from collections import deque
from tensorflow.keras.models import load_model


CASCADE_PATH = r'C:\Users\Lenovo\Downloads\archive\haarcascade_frontalface_default.xml'
MODEL_PATH   = r'C:\Users\Lenovo\Downloads\archive\emotion_model.hdf5'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_model = load_model(MODEL_PATH, compile=False)

try:
    _, H, W, C = emotion_model.input_shape
except Exception:
    dims = [d for d in emotion_model.input_shape if d is not None]
    if len(dims) == 3:
        H, W, C = dims
    else:
        H, W, C = 64, 64, 1  

print(f"Model expects input: {H}x{W}x{C}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


CAM_WIDTH = 640
CAM_HEIGHT = 480

alpha = 0.65
distance_threshold = 80
max_unseen_frames = 12
CONF_THRESH = 0.30   

trackers = {}
next_tracker_id = 0
frame_idx = 0


def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        i = idxs[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cam.isOpened():
    raise RuntimeError("Camera not accessible")

print("Camera started. Press 'q' to quit.")
fps_smooth = None
t_prev = time.time()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to read frame")
        break
    frame_idx += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = frame.shape[:2]

    
    detections = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Remove duplicates
    dets = non_max_suppression(detections, overlapThresh=0.4)

    assigned_tracker = {}
    used_trackers = set()

    for i, det in enumerate(dets):
        x, y, w, h = det
        cx, cy = x + w/2, y + h/2
        best_tid = None
        best_dist = float('inf')
        for tid, tr in trackers.items():
            tcx, tcy = tr['centroid']
            d = math.hypot(cx - tcx, cy - tcy)
            if d < best_dist:
                best_dist = d
                best_tid = tid
        if best_tid is not None and best_dist < distance_threshold and best_tid not in used_trackers:
            assigned_tracker[i] = best_tid
            used_trackers.add(best_tid)
        else:
            assigned_tracker[i] = None

    for i, det in enumerate(dets):
        x, y, w, h = det
        cx, cy = x + w/2, y + h/2
        tid = assigned_tracker[i]

        if tid is None:
            tid = next_tracker_id
            next_tracker_id += 1
            trackers[tid] = {
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'last_seen': frame_idx,
                'history': deque(maxlen=5)  
            }
        else:
            ox, oy, ow, oh = trackers[tid]['bbox']
            ocx, ocy = trackers[tid]['centroid']
            nx = int(alpha * ox + (1 - alpha) * x)
            ny = int(alpha * oy + (1 - alpha) * y)
            nw = int(alpha * ow + (1 - alpha) * w)
            nh = int(alpha * oh + (1 - alpha) * h)
            ncx = alpha * ocx + (1 - alpha) * cx
            ncy = alpha * ocy + (1 - alpha) * cy
            trackers[tid]['bbox'] = (nx, ny, nw, nh)
            trackers[tid]['centroid'] = (ncx, ncy)
            trackers[tid]['last_seen'] = frame_idx


    to_delete = [tid for tid, tr in trackers.items() if frame_idx - tr['last_seen'] > max_unseen_frames]
    for tid in to_delete:
        del trackers[tid]

    for tid, tr in trackers.items():
        x, y, w, h = tr['bbox']
        pad = int(0.25 * max(w, h))
        cx, cy = int(tr['centroid'][0]), int(tr['centroid'][1])
        half = max(w, h)//2 + pad
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(fw, cx + half)
        y2 = min(fh, cy + half)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue

        face_roi_gray = frame_gray[y1:y2, x1:x2]
        face_roi_color = frame[y1:y2, x1:x2]

        if face_roi_gray.size == 0:
            continue

        #Preprocessing the emotion model input image using the face ROI
        face_resized = cv2.resize(face_roi_gray, (W, H), interpolation=cv2.INTER_AREA)
        face_resized = cv2.equalizeHist(face_resized)
        face_resized = face_resized.astype("float32") / 127.5 - 1.0

        if C == 1:
            face_resized = np.expand_dims(face_resized, axis=-1)
        elif C == 3:
            face_resized = cv2.resize(face_roi_color, (W, H))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_resized = face_resized.astype("float32") / 127.5 - 1.0

        face_batch = np.expand_dims(face_resized, axis=0)

      
        preds = emotion_model.predict(face_batch, verbose=0)
        label_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        tr['history'].append(label_idx)
        label_idx = max(set(tr['history']), key=tr['history'].count)

        label_text = emotion_labels[label_idx] if confidence >= CONF_THRESH else "Neutral"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label_text} ({int(confidence*100)}%)"
        cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    t_now = time.time()
    fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
    t_prev = t_now
    fps_smooth = fps if fps_smooth is None else (0.85*fps_smooth + 0.15*fps)
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Stable Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
