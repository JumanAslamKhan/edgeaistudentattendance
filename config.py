SIMILARITY_THRESHOLD = 0.65
FACE_DISTANCE_THRESHOLD = 0.58
FACE_DISTANCE_MARGIN = 0.08
REGISTRATION_SAMPLES = 10
ATTENDANCE_WINDOW_SECONDS = 300
FRAME_SKIP = 2
FRAME_RESIZE_SCALE = 0.5
WEBCAM_INDEX = 1
DB_PATH = "attendance.db"

# Feature flags and thresholds
TRACK_UNKNOWN_FACES = True
CONFIDENCE_REVIEW_THRESHOLD = 0.50
ENABLE_LIVENESS_CHECK = True
BLINK_DETECTION_FRAMES = 5

YEARS    = ["FE", "SE", "TE", "BE"]
BRANCHES = ["COMPS", "AIDS", "CSD", "MTRX"]
DIVS     = ["A", "B", "C", "D"]
SUBJECTS = {
    "FE":   ["Engineering Maths", "Engineering Physics", "Engineering Chemistry", "BEE", "Engineering Mechanics"],
    "SE":   ["DSA", "DBMS", "OOP", "Discrete Maths", "Computer Networks"],
    "TE":   ["Machine Learning", "OS", "Theory of Computation", "CNS", "Software Engineering"],
    "BE":   ["Deep Learning", "Cloud Computing", "Big Data", "Project", "Distributed Systems"],
}
