import sqlite3
import numpy as np
import json
from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS professors (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            mobile   TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS professor_branches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            professor_id INTEGER NOT NULL,
            branch       TEXT NOT NULL,
            UNIQUE(professor_id, branch)
        );

        CREATE TABLE IF NOT EXISTS hods (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            mobile   TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            branch   TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS students (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ien      TEXT NOT NULL UNIQUE,
            name     TEXT NOT NULL,
            password TEXT NOT NULL,
            email    TEXT NOT NULL UNIQUE,
            mobile   TEXT NOT NULL,
            branch   TEXT NOT NULL,
            year     TEXT NOT NULL,
            div      TEXT,
            roll_no  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            role      TEXT NOT NULL CHECK(role IN ('professor', 'student')),
            embedding TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            professor_id INTEGER NOT NULL,
            branch       TEXT NOT NULL,
            year         TEXT NOT NULL,
            subject      TEXT NOT NULL,
            start_time   TEXT NOT NULL,
            end_time     TEXT
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            timestamp  TEXT NOT NULL,
            confidence REAL NOT NULL,
            status     TEXT NOT NULL DEFAULT 'present'
        );
    """)
    conn.commit()
    conn.close()


# ── Embeddings ────────────────────────────────────────────────────────────────

def _save_embedding(person_id, role, embedding, conn, c):
    c.execute("DELETE FROM embeddings WHERE person_id=? AND role=?", (person_id, role))
    c.execute("INSERT INTO embeddings (person_id, role, embedding) VALUES (?,?,?)",
              (person_id, role, json.dumps(embedding.tolist())))


# ── Professor ─────────────────────────────────────────────────────────────────

def register_professor(name, mobile, password, branches, embedding):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO professors (name, mobile, password) VALUES (?,?,?)",
                  (name, mobile, password))
        pid = c.lastrowid
        for b in branches:
            c.execute("INSERT OR IGNORE INTO professor_branches (professor_id, branch) VALUES (?,?)",
                      (pid, b))
        _save_embedding(pid, 'professor', embedding, conn, c)
        conn.commit()
        conn.close()
        return pid, None
    except sqlite3.IntegrityError as e:
        conn.close()
        return None, "Mobile number already registered."


def get_professor_branches(professor_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT branch FROM professor_branches WHERE professor_id=?", (professor_id,))
    branches = [r[0] for r in c.fetchall()]
    conn.close()
    return branches


# ── HOD ──────────────────────────────────────────────────────────────────────

def register_hod(name, mobile, password, branch):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO hods (name, mobile, password, branch) VALUES (?,?,?,?)",
                  (name, mobile, password, branch))
        conn.commit()
        hid = c.lastrowid
        conn.close()
        return hid, None
    except sqlite3.IntegrityError as e:
        conn.close()
        if "branch" in str(e):
            return None, f"HOD for {branch} already registered."
        return None, "Mobile number already registered."


def get_hod_by_mobile(mobile, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, branch FROM hods WHERE mobile=? AND password=?", (mobile, password))
    row = c.fetchone()
    conn.close()
    return row


# ── Student ───────────────────────────────────────────────────────────────────

def register_student(ien, name, password, email, mobile, branch, year, div, roll_no, embedding):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO students (ien, name, password, email, mobile, branch, year, div, roll_no)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (ien, name, password, email, mobile, branch, year, div, roll_no))
        sid = c.lastrowid
        _save_embedding(sid, 'student', embedding, conn, c)
        conn.commit()
        conn.close()
        return sid, None
    except sqlite3.IntegrityError as e:
        conn.close()
        if "ien"   in str(e): return None, "IEN already registered."
        if "email" in str(e): return None, "Email already registered."
        return None, str(e)


def get_students_by_branch_year(branch, year):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM students WHERE branch=? AND year=?", (branch, year))
    ids = {r[0] for r in c.fetchall()}
    conn.close()
    return ids


# ── Embeddings loader ─────────────────────────────────────────────────────────

def load_all_embeddings():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT e.person_id, e.role, e.embedding,
               COALESCE(p.name, s.name) as name
        FROM embeddings e
        LEFT JOIN professors p ON e.role='professor' AND e.person_id=p.id
        LEFT JOIN students   s ON e.role='student'   AND e.person_id=s.id
    """)
    rows = c.fetchall()
    conn.close()
    return [{"person_id": r[0], "role": r[1], "name": r[3],
             "embedding": np.array(json.loads(r[2]))} for r in rows]


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(professor_id, branch, year, subject, start_time):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions (professor_id, branch, year, subject, start_time)
        VALUES (?,?,?,?,?)
    """, (professor_id, branch, year, subject, start_time))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid


def close_session(session_id, end_time):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE sessions SET end_time=? WHERE id=?", (end_time, session_id))
    conn.commit()
    conn.close()


def mark_attendance(session_id, student_id, timestamp, confidence, status="present"):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO attendance (session_id, student_id, timestamp, confidence, status)
        VALUES (?,?,?,?,?)
    """, (session_id, student_id, timestamp, confidence, status))
    conn.commit()
    conn.close()


def mark_absent_students(session_id, present_ids, eligible_ids, end_time):
    conn = get_connection()
    c = conn.cursor()
    for sid in (eligible_ids - present_ids):
        c.execute("""
            INSERT INTO attendance (session_id, student_id, timestamp, confidence, status)
            VALUES (?,?,?,0.0,'absent')
        """, (session_id, sid, end_time))
    conn.commit()
    conn.close()


def update_attendance(session_id, student_id, status):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE attendance SET status=? WHERE session_id=? AND student_id=?
    """, (status, session_id, student_id))
    conn.commit()
    conn.close()


def delete_attendance(session_id, student_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE session_id=? AND student_id=?", 
              (session_id, student_id))
    conn.commit()
    conn.close()


def get_session_attendance(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT st.id, st.name, st.ien, st.roll_no, st.div, a.status, a.confidence, a.timestamp
        FROM students st
        JOIN sessions s ON s.id=?
        LEFT JOIN attendance a ON a.student_id=st.id AND a.session_id=s.id
        WHERE st.branch=s.branch AND st.year=s.year
        ORDER BY a.status DESC, CAST(st.roll_no AS INTEGER)
    """, (session_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_session_info(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT s.id, s.subject, s.branch, s.year, s.start_time, s.end_time, p.name
        FROM sessions s
        JOIN professors p ON s.professor_id=p.id
        WHERE s.id=?
    """, (session_id,))
    row = c.fetchone()
    conn.close()
    return row


# ── HOD report queries ────────────────────────────────────────────────────────

def hod_get_professors(branch):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.id, p.name
        FROM professors p
        JOIN professor_branches pb ON pb.professor_id=p.id
        WHERE pb.branch=?
        ORDER BY p.name
    """, (branch,))
    rows = c.fetchall()
    conn.close()
    return rows


def hod_get_students(branch):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, ien, name, year, div, roll_no
        FROM students WHERE branch=?
        ORDER BY year, CAST(roll_no AS INTEGER)
    """, (branch,))
    rows = c.fetchall()
    conn.close()
    return rows


def hod_get_sessions(branch, professor_id=None):
    conn = get_connection()
    c = conn.cursor()
    if professor_id:
        c.execute("""
            SELECT s.id, p.name, s.year, s.branch, s.subject, s.start_time,
                   COUNT(a.id) FILTER(WHERE a.status='present') as present,
                   COUNT(a.id) FILTER(WHERE a.status='absent')  as absent
            FROM sessions s
            JOIN professors p ON s.professor_id=p.id
            LEFT JOIN attendance a ON a.session_id=s.id
            WHERE s.branch=? AND s.professor_id=?
            GROUP BY s.id ORDER BY s.start_time DESC
        """, (branch, professor_id))
    else:
        c.execute("""
            SELECT s.id, p.name, s.year, s.branch, s.subject, s.start_time,
                   COUNT(a.id) FILTER(WHERE a.status='present') as present,
                   COUNT(a.id) FILTER(WHERE a.status='absent')  as absent
            FROM sessions s
            JOIN professors p ON s.professor_id=p.id
            LEFT JOIN attendance a ON a.session_id=s.id
            WHERE s.branch=?
            GROUP BY s.id ORDER BY s.start_time DESC
        """, (branch,))
    rows = c.fetchall()
    conn.close()
    return rows


def hod_student_attendance_summary(branch):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT st.name, st.year, st.div, st.roll_no,
               COUNT(a.id) FILTER(WHERE a.status='present') as present,
               COUNT(a.id) FILTER(WHERE a.status='absent')  as absent
        FROM students st
        LEFT JOIN attendance a ON a.student_id=st.id
        WHERE st.branch=?
        GROUP BY st.id
        ORDER BY st.year, CAST(st.roll_no AS INTEGER)
    """, (branch,))
    rows = c.fetchall()
    conn.close()
    return rows


# ── Student login & dashboard queries ─────────────────────────────────────────

def get_student_by_credentials(mobile, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, branch, year, div, roll_no, ien FROM students WHERE mobile=? AND password=?", (mobile, password))
    row = c.fetchone()
    conn.close()
    return row


def student_get_subjects(student_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT s.subject
        FROM attendance a
        JOIN sessions s ON a.session_id = s.id
        WHERE a.student_id = ?
        ORDER BY s.subject
    """, (student_id,))
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows


def student_get_attendance_summary(student_id, subject=None):
    conn = get_connection()
    c = conn.cursor()
    if subject and subject != 'all':
        c.execute("""
            SELECT
                COUNT(a.id) FILTER(WHERE a.status='present') as present,
                COUNT(a.id) FILTER(WHERE a.status='absent')  as absent,
                COUNT(a.id) as total
            FROM attendance a
            JOIN sessions s ON a.session_id = s.id
            WHERE a.student_id = ? AND s.subject = ?
        """, (student_id, subject))
    else:
        c.execute("""
            SELECT
                COUNT(a.id) FILTER(WHERE a.status='present') as present,
                COUNT(a.id) FILTER(WHERE a.status='absent')  as absent,
                COUNT(a.id) as total
            FROM attendance a
            WHERE a.student_id = ?
        """, (student_id,))
    row = c.fetchone()
    conn.close()
    return row


def student_get_subject_breakdown(student_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT s.subject,
               COUNT(a.id) FILTER(WHERE a.status='present') as present,
               COUNT(a.id) FILTER(WHERE a.status='absent')  as absent,
               COUNT(a.id) as total
        FROM attendance a
        JOIN sessions s ON a.session_id = s.id
        WHERE a.student_id = ?
        GROUP BY s.subject
        ORDER BY s.subject
    """, (student_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def student_get_session_history(student_id, subject=None):
    conn = get_connection()
    c = conn.cursor()
    if subject and subject != 'all':
        c.execute("""
            SELECT s.subject, s.branch, s.year, s.start_time,
                   a.status, a.confidence, p.name as professor
            FROM attendance a
            JOIN sessions s ON a.session_id = s.id
            JOIN professors p ON s.professor_id = p.id
            WHERE a.student_id = ? AND s.subject = ?
            ORDER BY s.start_time DESC
        """, (student_id, subject))
    else:
        c.execute("""
            SELECT s.subject, s.branch, s.year, s.start_time,
                   a.status, a.confidence, p.name as professor
            FROM attendance a
            JOIN sessions s ON a.session_id = s.id
            JOIN professors p ON s.professor_id = p.id
            WHERE a.student_id = ?
            ORDER BY s.start_time DESC
        """, (student_id,))
    rows = c.fetchall()
    conn.close()
    return rows
