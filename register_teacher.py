import cv2
import sys
import face_recognition
import numpy as np
from database import init_db, register_teacher
from config import WEBCAM_INDEX, REGISTRATION_SAMPLES


def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def main():
    init_db()
    name = input("Enter teacher name: ").strip()
    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    collected = []
    print(f"Registering teacher: {name}")
    print(f"Press SPACE to capture. Need {REGISTRATION_SAMPLES} samples. Q to quit.")

    while len(collected) < REGISTRATION_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        for (t, r, b, l) in locations:
            cv2.rectangle(display, (l, t), (r, b), (255, 180, 0), 2)
        cv2.putText(display, f"Teacher: {name}  [{len(collected)}/{REGISTRATION_SAMPLES}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)
        cv2.putText(display, "SPACE = capture  |  Q = quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Register Teacher", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(locations) == 1:
                encs = face_recognition.face_encodings(rgb, locations)
                if encs:
                    collected.append(encs[0])
                    print(f"Captured {len(collected)}/{REGISTRATION_SAMPLES}")
            elif len(locations) == 0:
                print("No face detected.")
            else:
                print("Multiple faces. Show only one.")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    tid = register_teacher(name, average_embeddings(collected))
    print(f"Teacher '{name}' registered with ID {tid}.")


if __name__ == "__main__":
    main()