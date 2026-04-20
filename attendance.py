from datetime import datetime
from database import (create_session, close_session, mark_attendance,
                      mark_absent_students, get_students_by_branch_year,
                      get_professor_branches)
from config import ATTENDANCE_WINDOW_SECONDS


class AttendanceManager:
    def __init__(self):
        self.pending_professor_id   = None
        self.pending_professor_name = None
        self.allowed_branches       = []
        self.branch  = None
        self.year    = None
        self.reset()

    def reset(self):
        self.session_id    = None
        self.session_start = None
        self.professor_id  = None
        self.professor_name = None
        self.branch        = None
        self.year          = None
        self.subject       = None
        self.present_ids   = set()
        self.eligible_ids  = set()
        self.active        = False

    def set_pending_professor(self, professor_id, professor_name):
        if not self.active:
            self.pending_professor_id   = professor_id
            self.pending_professor_name = professor_name
            self.allowed_branches       = get_professor_branches(professor_id)

    def start_session(self, branch, year, subject):
        if self.pending_professor_id is None:
            return False
        now = datetime.now().isoformat()
        self.session_id     = create_session(self.pending_professor_id, branch, year, subject, now)
        self.session_start  = datetime.now()
        self.professor_id   = self.pending_professor_id
        self.professor_name = self.pending_professor_name
        self.branch         = branch
        self.year           = year
        self.subject        = subject
        self.present_ids    = set()
        self.eligible_ids   = get_students_by_branch_year(branch, year)
        self.active         = True
        self.pending_professor_id   = None
        self.pending_professor_name = None
        self.allowed_branches       = []
        print(f"Session | {self.professor_name} | {year} {branch} | {subject} | "
              f"Eligible: {len(self.eligible_ids)}")
        return True

    def is_window_open(self):
        if not self.active or not self.session_start:
            return False
        return (datetime.now() - self.session_start).total_seconds() <= ATTENDANCE_WINDOW_SECONDS

    def record_student(self, student_id, student_name, confidence):
        if student_id not in self.eligible_ids:
            return False, "not_eligible"
        if student_id in self.present_ids:
            return False, "duplicate"
        mark_attendance(self.session_id, student_id,
                        datetime.now().isoformat(), confidence)
        self.present_ids.add(student_id)
        print(f"PRESENT: {student_name} ({confidence:.2f})")
        return True, "marked"

    def end_session(self):
        if not self.active:
            return
        end_time = datetime.now().isoformat()
        mark_absent_students(self.session_id, self.present_ids,
                             self.eligible_ids, end_time)
        close_session(self.session_id, end_time)
        print(f"Session {self.session_id} closed. "
              f"Present: {len(self.present_ids)}/{len(self.eligible_ids)}")
        self.reset()

    def remaining_seconds(self):
        if not self.session_start:
            return 0
        return max(0, int(ATTENDANCE_WINDOW_SECONDS -
                          (datetime.now() - self.session_start).total_seconds()))
