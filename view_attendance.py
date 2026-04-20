import sqlite3
from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def list_sessions():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT s.id, t.name, s.year, s.branch, s.subject, s.start_time,
               COUNT(a.id) FILTER (WHERE a.status='present') as present_count,
               COUNT(a.id) FILTER (WHERE a.status='absent')  as absent_count
        FROM sessions s
        LEFT JOIN teachers t ON s.teacher_id = t.id
        LEFT JOIN attendance a ON a.session_id = s.id
        GROUP BY s.id
        ORDER BY s.id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def get_session_attendance(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT st.name, st.roll_no, st.div, a.status, a.confidence, a.timestamp
        FROM attendance a
        JOIN students st ON a.student_id = st.id
        WHERE a.session_id = ?
        ORDER BY a.status DESC, CAST(st.roll_no AS INTEGER)
    """, (session_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def sep(w=70): print("-" * w)


def main():
    print("\n" + "=" * 70)
    print("         ATTENDANCE SYSTEM  —  REPORT VIEWER")
    print("=" * 70)

    sessions = list_sessions()
    if not sessions:
        print("No sessions found.")
        return

    print(f"\n{'ID':<5} {'Teacher':<14} {'Year':<5} {'Branch':<7} {'Subject':<22} {'Date':<18} P / A")
    sep()
    for sid, teacher, year, branch, subject, start, present, absent in sessions:
        date = str(start)[:16].replace("T", " ")
        subj = str(subject)[:20]
        print(f"{sid:<5} {str(teacher):<14} {str(year):<5} {str(branch):<7} {subj:<22} {date:<18} {present} / {absent}")

    sep()
    print("\nEnter session ID for details  (0 to exit): ", end="")

    while True:
        try:
            choice = int(input())
        except ValueError:
            print("Enter a valid number: ", end="")
            continue

        if choice == 0:
            break

        session_info = next((s for s in sessions if s[0] == choice), None)
        if not session_info:
            print("Session not found.")
            print("Enter session ID (0 to exit): ", end="")
            continue

        sid, teacher, year, branch, subject, start, present, absent = session_info
        print(f"\n{'='*70}")
        print(f"  Session {sid}  |  {teacher}  |  {year} {branch}  |  {subject}")
        print(f"  Date    : {str(start)[:19].replace('T', ' ')}")
        print(f"  Present : {present}   Absent: {absent}   Total: {present + absent}")
        print(f"{'='*70}")
        print(f"{'Name':<22} {'Roll':<6} {'Div':<5} {'Status':<12} {'Conf':<8} Time")
        sep()

        records = get_session_attendance(choice)
        for name, roll, div, status, conf, ts in records:
            time_str = str(ts)[:16].replace("T", " ") if ts else "—"
            conf_str = f"{conf:.2f}" if conf and conf > 0 else "—"
            div_str  = str(div) if div else "—"
            mark     = "✓ PRESENT" if status == "present" else "✗ ABSENT"
            print(f"{str(name):<22} {str(roll):<6} {div_str:<5} {mark:<12} {conf_str:<8} {time_str}")

        sep()
        print("\nEnter session ID (0 to exit): ", end="")


if __name__ == "__main__":
    main()