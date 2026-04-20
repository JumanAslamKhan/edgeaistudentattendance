import cv2
import sys
import face_recognition
from database import init_db, register_person
from face_utils import average_embeddings
from config import REGISTRATION_SAMPLES, WEBCAM_INDEX


def register(name, role):
    init_db()
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    collected = []
    print(f"Registering {role}: {name}")
    print(f"Press SPACE to capture. Need {REGISTRATION_SAMPLES} samples. Press Q to quit.")

    while len(collected) < REGISTRATION_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        for (t, r, b, l) in locations:
            cv2.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

        cv2.putText(display, f"Samples: {len(collected)}/{REGISTRATION_SAMPLES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Register - SPACE to capture, Q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(locations) == 1:
                encodings = face_recognition.face_encodings(rgb, locations)
                if encodings:
                    collected.append(encodings[0])
                    print(f"Captured {len(collected)}/{REGISTRATION_SAMPLES}")
            elif len(locations) == 0:
                print("No face detected.")
            else:
                print("Multiple faces detected. Show only one face.")
        elif key == ord('q'):
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    avg = average_embeddings(collected)
    pid = register_person(name, role, avg)
    print(f"Registered {role} '{name}' with ID {pid}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python register.py <name> <teacher|student>")
        sys.exit(1)
    role = sys.argv[2].lower()
    if role not in ("teacher", "student"):
        print("Role must be 'teacher' or 'student'.")
        sys.exit(1)
    register(sys.argv[1], role)
