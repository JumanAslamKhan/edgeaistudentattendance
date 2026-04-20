import cv2
import sys
import base64
import numpy as np
import face_recognition
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from database import init_db, register_student
from config import YEARS, BRANCHES, DIVS, REGISTRATION_SAMPLES

app = Flask(__name__)
init_db()

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Student Registration</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { width: 100%; max-width: 860px; background: #1a1d27; border-radius: 12px; padding: 36px; box-shadow: 0 4px 30px rgba(0,0,0,0.5); }
  h2 { color: #00d4ff; margin-bottom: 24px; font-size: 1.6rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .full { grid-column: 1 / -1; }
  label { font-size: 0.8rem; color: #aaa; margin-bottom: 4px; display: block; }
  input, select { width: 100%; padding: 10px 12px; background: #0f1117; border: 1px solid #333; border-radius: 6px; color: #fff; font-size: 0.95rem; }
  input:focus, select:focus { outline: none; border-color: #00d4ff; }
  .cam-section { margin-top: 24px; }
  .cam-section h3 { color: #00d4ff; margin-bottom: 12px; }
  .cam-wrap { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  video, canvas { border-radius: 8px; border: 2px solid #333; background: #000; }
  video { width: 320px; height: 240px; }
  canvas { display: none; }
  .progress-bar { background: #0f1117; border-radius: 20px; overflow: hidden; height: 14px; margin-top: 10px; border: 1px solid #333; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #0078ff); transition: width 0.3s; }
  .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95rem; font-weight: bold; }
  .btn-capture { background: #00d4ff; color: #000; }
  .btn-capture:disabled { background: #333; color: #666; cursor: not-allowed; }
  .btn-submit { background: #00c853; color: #000; width: 100%; margin-top: 20px; padding: 14px; font-size: 1rem; }
  .btn-submit:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 10px; font-size: 0.9rem; min-height: 22px; }
  .ok { color: #00e676; }
  .err { color: #ff5252; }
  .thumb-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; max-width: 320px; }
  .thumb-wrap img { width: 58px; height: 58px; border-radius: 4px; border: 2px solid #00d4ff; object-fit: cover; }
  .alert { padding: 14px; border-radius: 8px; margin-bottom: 20px; font-size: 0.95rem; }
  .alert-ok  { background: #0a2e1a; border: 1px solid #00c853; color: #00e676; }
  .alert-err { background: #2e0a0a; border: 1px solid #ff5252; color: #ff5252; }
</style>
</head>
<body>
<div class="container">
  <h2>🎓 Student Registration</h2>

  {% if message %}
  <div class="alert {{ 'alert-ok' if success else 'alert-err' }}">{{ message }}</div>
  {% endif %}

  <form id="regForm">
    <div class="grid">
      <div>
        <label>IEN (Institute Enrollment No.)</label>
        <input type="text" name="ien" required placeholder="e.g. 2024COMPS001">
      </div>
      <div>
        <label>Full Name</label>
        <input type="text" name="name" required placeholder="e.g. Alice Sharma">
      </div>
      <div>
        <label>Email ID</label>
        <input type="email" name="email" required placeholder="alice@college.edu">
      </div>
      <div>
        <label>Mobile Number</label>
        <input type="tel" name="mobile" required placeholder="9876543210">
      </div>
      <div>
        <label>Password</label>
        <input type="password" name="password" required placeholder="Min 6 characters">
      </div>
      <div>
        <label>Roll Number</label>
        <input type="text" name="roll_no" required placeholder="e.g. 42">
      </div>
      <div>
        <label>Branch</label>
        <select name="branch" required>
          <option value="">-- Select Branch --</option>
          {% for b in branches %}<option value="{{ b }}">{{ b }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Year</label>
        <select name="year" required>
          <option value="">-- Select Year --</option>
          {% for y in years %}<option value="{{ y }}">{{ y }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Division (optional)</label>
        <select name="div">
          <option value="">-- No Division --</option>
          {% for d in divs %}<option value="{{ d }}">{{ d }}</option>{% endfor %}
        </select>
      </div>
    </div>

    <div class="cam-section">
      <h3>📸 Face Registration ({{ samples }} photos required)</h3>
      <div class="cam-wrap">
        <div>
          <video id="video" autoplay playsinline></video><br>
          <button type="button" class="btn btn-capture" id="captureBtn" onclick="capture()" style="margin-top:10px;width:100%">
            📷 Capture Photo
          </button>
          <div class="progress-bar" style="margin-top:8px">
            <div class="progress-fill" id="progress" style="width:0%"></div>
          </div>
          <div class="status" id="camStatus">0 / {{ samples }} captured</div>
        </div>
        <div>
          <div class="thumb-wrap" id="thumbs"></div>
        </div>
      </div>
    </div>

    <input type="hidden" name="images" id="imagesField">
    <button type="button" class="btn btn-submit" id="submitBtn" onclick="submitForm()" disabled>
      ✅ Register Student
    </button>
    <div class="status" id="submitStatus"></div>
  </form>
</div>

<script>
const SAMPLES = {{ samples }};
let captured = [];

navigator.mediaDevices.getUserMedia({ video: true })
  .then(s => document.getElementById('video').srcObject = s)
  .catch(() => document.getElementById('camStatus').textContent = 'Camera not available.');

function capture() {
  const video = document.getElementById('video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  captured.push(dataUrl);

  const thumb = document.createElement('img');
  thumb.src = dataUrl;
  document.getElementById('thumbs').appendChild(thumb);

  const pct = (captured.length / SAMPLES) * 100;
  document.getElementById('progress').style.width = pct + '%';
  document.getElementById('camStatus').textContent = `${captured.length} / ${SAMPLES} captured`;

  if (captured.length >= SAMPLES) {
    document.getElementById('captureBtn').disabled = true;
    document.getElementById('submitBtn').disabled = false;
    document.getElementById('camStatus').className = 'status ok';
    document.getElementById('camStatus').textContent = '✓ All photos captured. Ready to register.';
  }
}

async function submitForm() {
  const form = document.getElementById('regForm');
  const data = new FormData(form);
  const obj = {};
  for (const [k, v] of data.entries()) obj[k] = v;
  obj.images = captured;

  document.getElementById('submitBtn').disabled = true;
  document.getElementById('submitStatus').textContent = 'Registering...';
  document.getElementById('submitStatus').className = 'status';

  const res = await fetch('/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });
  const result = await res.json();

  if (result.success) {
    document.getElementById('submitStatus').className = 'status ok';
    document.getElementById('submitStatus').textContent = '✓ ' + result.message;
    setTimeout(() => location.reload(), 2000);
  } else {
    document.getElementById('submitStatus').className = 'status err';
    document.getElementById('submitStatus').textContent = '✗ ' + result.message;
    document.getElementById('submitBtn').disabled = false;
  }
}
</script>
</body>
</html>
"""


def extract_embedding_from_b64(b64_str):
    header, data = b64_str.split(',', 1)
    img_bytes = base64.b64decode(data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    encs = face_recognition.face_encodings(rgb, locations)
    return encs[0] if encs else None


@app.route('/')
def index():
    return render_template_string(HTML,
                                  branches=BRANCHES,
                                  years=YEARS,
                                  divs=DIVS,
                                  samples=REGISTRATION_SAMPLES,
                                  message=None,
                                  success=False)


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required = ['ien', 'name', 'password', 'email', 'mobile', 'branch', 'year', 'roll_no', 'images']
    for field in required:
        if not data.get(field):
            return jsonify(success=False, message=f"Missing field: {field}")

    if len(data['password']) < 6:
        return jsonify(success=False, message="Password must be at least 6 characters.")

    images = data.get('images', [])
    if len(images) < REGISTRATION_SAMPLES:
        return jsonify(success=False, message=f"Need {REGISTRATION_SAMPLES} photos, got {len(images)}.")

    embeddings = []
    for b64 in images:
        emb = extract_embedding_from_b64(b64)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) < 5:
        return jsonify(success=False, message=f"Could only detect faces in {len(embeddings)} photos. Please retake in good lighting.")

    avg_embedding = np.mean(embeddings, axis=0)

    sid, error = register_student(
        ien=data['ien'],
        name=data['name'],
        password=data['password'],
        email=data['email'],
        mobile=data['mobile'],
        branch=data['branch'],
        year=data['year'],
        div=data.get('div', ''),
        roll_no=data['roll_no'],
        embedding=avg_embedding
    )

    if error:
        return jsonify(success=False, message=error)

    return jsonify(success=True, message=f"Student '{data['name']}' registered successfully! (ID: {sid})")


if __name__ == '__main__':
    print("Student Registration Portal running at http://localhost:5000")
    print("Press Ctrl+C to stop.")
    app.run(debug=False, port=5000)