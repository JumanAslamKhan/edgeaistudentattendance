import os
from contextlib import contextmanager

if "OPENCV_LOG_LEVEL" not in os.environ:
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import time
import sys
from database import init_db, load_all_embeddings
from face_utils import detect_and_encode, match_face, is_face_live
from attendance import AttendanceManager
from config import WEBCAM_INDEX, FRAME_SKIP, YEARS, BRANCHES, SUBJECTS, TRACK_UNKNOWN_FACES, CONFIDENCE_REVIEW_THRESHOLD

STATE_WAITING        = "waiting"
STATE_SELECT_BRANCH  = "select_branch"
STATE_SELECT_YEAR    = "select_year"
STATE_SELECT_SUBJECT = "select_subject"
STATE_ACTIVE         = "active"
STATE_COOLDOWN       = "cooldown"

COOLDOWN_SECONDS = 15


@contextmanager
def suppress_console_noise_on_windows():
    if not sys.platform.startswith("win"):
        yield
        return

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(null_fd)


def draw_menu(frame, title, options):
    h, w = frame.shape[:2]
    box_h = 60 + len(options) * 42
    y0 = h // 2 - box_h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 230, y0), (w//2 + 230, y0 + box_h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (w//2 - 230, y0), (w//2 + 230, y0 + box_h), (0, 180, 220), 2)
    cv2.putText(frame, title, (w//2 - 210, y0 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 210, 255), 2)
    for i, opt in enumerate(options):
        cv2.putText(frame, f"  {i+1}.  {opt}",
                    (w//2 - 210, y0 + 75 + i * 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


def draw_box(frame, loc, name, role, conf, marked=False, needs_review=False, is_unknown=False):
    t, r, b, l = loc
    thickness = 2
    
    # Determine color based on state
    if is_unknown:
        color = (0, 0, 255)  # Red for unknown
        thickness = 3  # Thicker border for unknown
    elif name == "Unknown":
        color = (0, 0, 255)  # Red for unknown
    elif role == 'professor':
        color = (255, 180, 0)  # Orange for professor
    elif marked:
        color = (0, 255, 80)  # Green for confirmed student
    elif needs_review:
        color = (0, 215, 255)  # Yellow for review needed
    elif role == "student":
        color = (0, 180, 255)  # Blue for student
    else:
        color = (80, 80, 80)  # Gray for default
    
    cv2.rectangle(frame, (l, t), (r, b), color, thickness)
    
    # Build label
    if needs_review:
        label = f"{name} {conf:.2f} [?]"
    elif is_unknown:
        label = "UNKNOWN"
    else:
        label = f"{name} {conf:.2f}" if conf > 0 else name
    
    cv2.putText(frame, label, (l, t - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_status(frame, state, mgr, cooldown_left=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (15, 15, 25), -1)
    if state == STATE_WAITING:
        msg, color = "Show professor face to begin", (180, 180, 180)
    elif state == STATE_SELECT_BRANCH:
        msg, color = f"Professor: {mgr.pending_professor_name}  |  Select branch (1-{len(BRANCHES)})", (0, 210, 255)
    elif state == STATE_SELECT_YEAR:
        msg, color = f"Branch: {mgr.branch}  |  Select year (1-{len(YEARS)})", (0, 210, 255)
    elif state == STATE_SELECT_SUBJECT:
        subs = SUBJECTS.get(mgr.year, [])
        msg, color = f"{mgr.year} {mgr.branch}  |  Select subject (1-{len(subs)})", (0, 210, 255)
    elif state == STATE_ACTIVE:
        rem = mgr.remaining_seconds()
        if rem > 0:
            msg = (f"SESSION | {mgr.professor_name} | {mgr.year} {mgr.branch} | "
                   f"{mgr.subject} | {rem}s | E=end Q=quit")
            color = (0, 220, 80)
        else:
            msg, color = "WINDOW CLOSED | Press E to finalise session", (0, 100, 255)
    elif state == STATE_COOLDOWN:
        msg, color = f"Session saved. New session in {cooldown_left}s...", (0, 165, 255)
    else:
        msg, color = "", (180, 180, 180)
    cv2.putText(frame, msg, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)


def main():
    init_db()
    known = load_all_embeddings()
    if not known:
        print("No registered faces. Register professor first: python register_professor.py")
        sys.exit(1)

    mgr = AttendanceManager()
    if sys.platform.startswith("win"):
        with suppress_console_noise_on_windows():
            cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    state          = STATE_WAITING
    selected_branch = None
    selected_year   = None
    frame_count    = 0
    fps_start      = time.time()
    fps            = 0.0
    last_results   = []
    confirmed      = {}
    cooldown_start = 0
    unknown_faces_seen = {}  # Track unknown faces for alerting

    print("Running. Q=quit | E=end session")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('e') and state == STATE_ACTIVE:
            mgr.end_session()
            state = STATE_COOLDOWN
            cooldown_start = time.time()

        # --- Branch selection ---
        if state == STATE_SELECT_BRANCH:
            for i, _ in enumerate(BRANCHES):
                if key == ord(str(i + 1)):
                    selected_branch = BRANCHES[i]
                    mgr.branch = selected_branch
                    state = STATE_SELECT_YEAR
                    break

        # --- Year selection ---
        elif state == STATE_SELECT_YEAR:
            for i, _ in enumerate(YEARS):
                if key == ord(str(i + 1)):
                    selected_year = YEARS[i]
                    mgr.year = selected_year
                    state = STATE_SELECT_SUBJECT
                    break

        # --- Subject selection ---
        elif state == STATE_SELECT_SUBJECT:
            subs = SUBJECTS.get(selected_year, [])
            for i, _ in enumerate(subs):
                if key == ord(str(i + 1)):
                    mgr.start_session(selected_branch, selected_year, subs[i])
                    state = STATE_ACTIVE
                    break

        # --- Cooldown ---
        cooldown_left = 0
        if state == STATE_COOLDOWN:
            cooldown_left = max(0, int(COOLDOWN_SECONDS - (time.time() - cooldown_start)))
            if cooldown_left == 0:
                state = STATE_WAITING
                mgr.pending_professor_id   = None
                mgr.pending_professor_name = None

        # --- Face detection ---
        if frame_count % FRAME_SKIP == 0:
            locations, encodings = detect_and_encode(frame)
            last_results = []
            for loc, enc in zip(locations, encodings):
                pid, name, role, conf = match_face(enc, known)
                marked = False
                needs_review = False
                is_unknown = False
                
                # Check for unknown faces
                if pid is None and TRACK_UNKNOWN_FACES:
                    is_unknown = True
                    unknown_faces_seen[time.time()] = {'name': name, 'conf': conf}
                
                # Check if confidence needs review
                if pid is not None and conf < CONFIDENCE_REVIEW_THRESHOLD and conf > 0:
                    needs_review = True

                if state == STATE_WAITING and role == 'professor' and pid:
                    mgr.set_pending_professor(pid, name)
                    state = STATE_SELECT_BRANCH

                elif state == STATE_ACTIVE and role == "student" and pid:
                    if mgr.is_window_open():
                        # Only auto-mark if confidence is high or review is approved
                        if not needs_review:
                            ok, reason = mgr.record_student(pid, name, conf)
                            if ok:
                                confirmed[pid] = time.time()
                                marked = True
                    else:
                        mgr.end_session()
                        state = STATE_COOLDOWN
                        cooldown_start = time.time()

                last_results.append((loc, name, role, conf,
                                     confirmed.get(pid, 0) > time.time() - 3 if pid else False,
                                     needs_review, is_unknown))

        # --- Draw ---
        for (loc, name, role, conf, flash, needs_review, is_unknown) in last_results:
            draw_box(frame, loc, name, role, conf, flash, needs_review, is_unknown)

        if state == STATE_SELECT_BRANCH:
            draw_menu(frame, "Select Branch:", BRANCHES)
        elif state == STATE_SELECT_YEAR:
            draw_menu(frame, f"Select Year ({selected_branch}):", YEARS)
        elif state == STATE_SELECT_SUBJECT:
            subs = SUBJECTS.get(selected_year, [])
            draw_menu(frame, f"Select Subject ({selected_year} {selected_branch}):", subs)

        draw_status(frame, state, mgr, cooldown_left)

        if frame_count % 30 == 0:
            fps = 30.0 / (time.time() - fps_start)
            fps_start = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)

        cv2.imshow("Attendance System", frame)

    if mgr.active:
        mgr.end_session()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
