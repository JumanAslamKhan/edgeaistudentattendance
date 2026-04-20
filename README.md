# Local Edge AI Face Recognition Attendance System

No cloud. No external APIs. Runs entirely on CPU.

---

## Requirements

- Python 3.10+
- Webcam

### Install build tools

**Ubuntu/Debian:**
```bash
sudo apt install cmake build-essential
```

**macOS:**
```bash
brew install cmake
```

**Windows:**
Install [CMake](https://cmake.org) and Visual Studio Build Tools.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Register teacher
```bash
python register.py "Dr. Smith" teacher
```

### 2. Register students
```bash
python register.py "Alice" student
python register.py "Bob" student
```
Press **SPACE** to capture each sample (30 needed). Press **Q** to cancel.

### 3. Run the system
```bash
python main.py
```

---

## Cloud Hosting

This project is ready to run in a Docker-based cloud host such as Render, Railway, Fly.io, or Docker on your own VM.

### Build and run locally with Docker

```bash
docker build -t studentattendanceedgeai .
docker run -p 8000:8000 studentattendanceedgeai
```

Open `http://localhost:8000` after the container starts.

### Cloud notes

- The web app entrypoint is `web_app:app`.
- The container listens on port `8000`.
- Use HTTPS on the deployed site so browser camera access works on mobile.
- The cloud build uses `requirements.cloud.txt`, which swaps in `opencv-python-headless` and `gunicorn` for server hosting.
- If Render fails with "Ran out of memory" while building wheels, redeploy with this Docker setup (Python 3.10 + binary `dlib`) to avoid source compilation.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `E` | End session manually |

---

## How It Works

1. Teacher face detected → session starts automatically
2. Attendance window opens for **5 minutes**
3. Students detected above threshold → marked **present**
4. Window closes → remaining students marked **absent**
5. All records saved to `attendance.db`

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.72 | Min cosine similarity to accept a match |
| `REGISTRATION_SAMPLES` | 30 | Face samples captured per person |
| `ATTENDANCE_WINDOW_SECONDS` | 300 | Seconds attendance window stays open |
| `FRAME_SKIP` | 2 | Process every Nth frame (performance) |
| `FRAME_RESIZE_SCALE` | 0.5 | Downscale factor before detection |
| `WEBCAM_INDEX` | 0 | Webcam device index |

---

## Database (`attendance.db`)

| Table | Purpose |
|---|---|
| `students` | Student identity records |
| `teachers` | Teacher identity records |
| `embeddings` | Averaged 128-dim face vectors (no images stored) |
| `sessions` | Teacher-initiated sessions with timestamps |
| `attendance` | Per-student records with confidence and status |
