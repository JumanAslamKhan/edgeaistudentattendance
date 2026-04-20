import cv2
import sys
import face_recognition
import numpy as np
from database import init_db, register_professor
from config import WEBCAM_INDEX, REGISTRATION_SAMPLES, BRANCHES


def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def main():
    init_db()
    print("=== Professor Registration ===")
    name = input("Full name         : ").strip()
    mobile = input("Mobile number     : ").strip()
    password = input("Password (min 6)  : ").strip()

    if not name or not mobile or len(password) < 6:
        print("Invalid input. Name and mobile required. Password min 6 chars.")
        sys.exit(1)

    print(f"\nAvailable branches: {', '.join(f'{i+1}={b}' for i,b in enumerate(BRANCHES))}")
    print("Enter branch numbers (comma separated, e.g. 1,3 for COMPS and CSD): ", end="")
    raw = input().strip()
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        branches = [BRANCHES[i] for i in indices]
    except (ValueError, IndexError):
        print("Invalid branch selection.")
        sys.exit(1)

    print(f"\nBranches assigned: {', '.join(branches)}")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    collected = []
    print(f"\nFace registration — press SPACE to capture ({REGISTRATION_SAMPLES} needed). Q to quit.")

    while len(collected) < REGISTRATION_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        for (t, r, b, l) in locations:
            cv2.rectangle(display, (l, t), (r, b), (255, 180, 0), 2)
        cv2.putText(display, f"Prof: {name}  [{len(collected)}/{REGISTRATION_SAMPLES}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 0), 2)
        cv2.putText(display, f"Branches: {', '.join(branches)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, "SPACE = capture  |  Q = quit",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        cv2.imshow("Register Professor", display)

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
                print("Multiple faces detected. Show only one.")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    pid, error = register_professor(name, mobile, password, branches,
                                    average_embeddings(collected))
    if error:
        print(f"Error: {error}")
    else:
        print(f"\nProfessor '{name}' registered with ID {pid}.")
        print(f"Branches: {', '.join(branches)}")


if __name__ == "__main__":
    main()
