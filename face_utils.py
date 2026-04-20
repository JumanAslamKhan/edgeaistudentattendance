import numpy as np
import face_recognition
import cv2
from config import FACE_DISTANCE_THRESHOLD, FACE_DISTANCE_MARGIN, FRAME_RESIZE_SCALE, ENABLE_LIVENESS_CHECK, BLINK_DETECTION_FRAMES


def detect_and_encode(frame):
    small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)
    scale = 1.0 / FRAME_RESIZE_SCALE
    scaled_locations = [
        (int(t * scale), int(r * scale), int(b * scale), int(l * scale))
        for (t, r, b, l) in locations
    ]
    return scaled_locations, encodings


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def match_face(encoding, known_embeddings):
    if not known_embeddings:
        return None, "Unknown", None, 0.0

    distances = [
        float(face_recognition.face_distance([entry["embedding"]], encoding)[0])
        for entry in known_embeddings
    ]
    ranked = sorted(zip(distances, known_embeddings), key=lambda item: item[0])
    best_distance, best_entry = ranked[0]
    second_best_distance = ranked[1][0] if len(ranked) > 1 else None

    if best_distance > FACE_DISTANCE_THRESHOLD:
        return None, "Unknown", None, 1.0 - best_distance

    if second_best_distance is not None and (second_best_distance - best_distance) < FACE_DISTANCE_MARGIN:
        return None, "Unknown", None, 1.0 - best_distance

    confidence = max(0.0, 1.0 - best_distance)
    return best_entry["person_id"], best_entry["name"], best_entry["role"], confidence


def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def detect_landmarks(image, face_locations):
    """Detect facial landmarks for liveness checking"""
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dlib_rects = detector(gray, 1)
    if not dlib_rects:
        return None
    
    landmarks = []
    for rect in dlib_rects:
        shape = predictor(gray, rect)
        landmarks.append(shape)
    return landmarks


def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio for blink detection"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


def is_face_live(frame, face_location, landmark_history):
    """Simple liveness check using eye blinking detection"""
    if not ENABLE_LIVENESS_CHECK:
        return True
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = face_recognition.face_landmarks(rgb, [face_location])
        if not landmarks:
            return False
        
        eyes = landmarks[0].get('left_eye', []) + landmarks[0].get('right_eye', [])
        if len(eyes) < 6:
            return False
        
        return True
    except:
        return False
