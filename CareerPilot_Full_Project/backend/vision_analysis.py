"""Approximate visible face/head geometry. Not gaze, emotion, or confidence detection."""
import cv2
import numpy as np

def analyze_frame(image_bytes: bytes) -> dict:
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("MediaPipe is unavailable; install requirements.txt") from exc
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode camera frame")
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)
    try:
        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        mesh.close()
    if not result.multi_face_landmarks:
        return {"face_detected": False, "head_orientation": "unknown", "note": "No face detected in this frame."}
    pts = result.multi_face_landmarks[0].landmark
    nose, left, right = pts[1], pts[234], pts[454]
    width = max(abs(right.x-left.x), 1e-5)
    ratio = (nose.x-(left.x+right.x)/2)/width
    orientation = "left" if ratio < -0.10 else "right" if ratio > 0.10 else "center"
    return {"face_detected": True, "head_orientation": orientation,
            "face_center_x": round(float(nose.x), 3), "face_center_y": round(float(nose.y), 3),
            "note": "Approximate head geometry only; not true gaze tracking or a measure of confidence, emotion, or competence."}
