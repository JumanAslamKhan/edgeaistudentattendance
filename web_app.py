import cv2
import sys
import base64
import csv
import io
from datetime import datetime
import numpy as np
import face_recognition
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, Response
from database import (init_db, register_student, register_hod, register_professor,
                      get_hod_by_mobile, get_professor_branches,
                      hod_get_professors, hod_get_students,
                      hod_get_sessions, hod_student_attendance_summary,
                      get_session_attendance, get_session_info,
                      update_attendance, delete_attendance, mark_attendance,
                      create_session, close_session, get_students_by_branch_year,
                      load_all_embeddings, mark_absent_students, get_connection,
                      get_student_by_credentials, student_get_subjects,
                      student_get_attendance_summary, student_get_subject_breakdown,
                      student_get_session_history)
from face_utils import match_face
from config import YEARS, BRANCHES, DIVS, REGISTRATION_SAMPLES, SUBJECTS

app = Flask(__name__)
app.secret_key = "attendance_secret_2024"
init_db()

# ─────────────────────────────────────────────────────────────────────────────
BASE_STYLE = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0f1a;color:#e0e0e0;min-height:100vh}
a{color:#00d4ff;text-decoration:none}a:hover{text-decoration:underline}
.topbar{background:#111827;padding:14px 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1f2937}
.topbar h1{color:#00d4ff;font-size:1.2rem;letter-spacing:1px}
.topbar nav a{margin-left:18px;color:#aaa;font-size:.88rem;padding:6px 12px;border-radius:5px}
.topbar nav a:hover{color:#00d4ff;background:#1f2937}
.topbar nav a.active{color:#00d4ff;border-bottom:2px solid #00d4ff}
.container{max-width:980px;margin:36px auto;padding:0 20px}
.card{background:#1a1d27;border-radius:12px;padding:32px;box-shadow:0 4px 24px rgba(0,0,0,.4);margin-bottom:24px}
h2{color:#00d4ff;margin-bottom:20px;font-size:1.4rem}
h3{color:#aad4ff;margin-bottom:14px;font-size:1.05rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.full{grid-column:1/-1}
label{font-size:.77rem;color:#888;margin-bottom:4px;display:block;text-transform:uppercase;letter-spacing:.4px}
input,select{width:100%;padding:10px 13px;background:#0d0f1a;border:1px solid #2d3244;border-radius:7px;color:#fff;font-size:.92rem}
input:focus,select:focus{outline:none;border-color:#00d4ff}
.btn{padding:10px 22px;border:none;border-radius:7px;cursor:pointer;font-size:.92rem;font-weight:600;transition:opacity .2s}
.btn:hover{opacity:.85}.btn:disabled{background:#2d3244;color:#666;cursor:not-allowed}
.btn-primary{background:#00d4ff;color:#000}
.btn-success{background:#00c853;color:#000}
.btn-danger{background:#ff5252;color:#fff}
.btn-warn{background:#ff9800;color:#000}
.btn-info{background:#00acc1;color:#000}
.btn-full{width:100%;padding:13px;font-size:1rem;margin-top:16px}
.alert{padding:13px 18px;border-radius:8px;margin-bottom:18px;font-size:.9rem}
.alert-ok{background:#0a2e1a;border:1px solid #00c853;color:#00e676}
.alert-err{background:#2e0a0a;border:1px solid #ff5252;color:#ff5252}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.74rem;font-weight:700}
.badge-present{background:#0a2e1a;color:#00e676;border:1px solid #00c853}
.badge-absent{background:#2e0a0a;color:#ff5252;border:1px solid #ff5252}
.badge-branch{background:#0a1f2e;color:#00d4ff;border:1px solid #00d4ff}
table{width:100%;border-collapse:collapse;font-size:.87rem}
th{background:#111827;color:#aaa;padding:10px 12px;text-align:left;font-weight:600;text-transform:uppercase;font-size:.74rem;letter-spacing:.4px}
td{padding:10px 12px;border-bottom:1px solid #1f2937}
tr:hover td{background:#1f2937}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px}
.stat-card{background:#111827;border-radius:10px;padding:20px;text-align:center;border:1px solid #1f2937}
.stat-card .num{font-size:2rem;font-weight:700;color:#00d4ff}
.stat-card .lbl{font-size:.76rem;color:#888;margin-top:4px;text-transform:uppercase}
.tabs{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap}
.tab{padding:8px 20px;border-radius:6px;cursor:pointer;font-size:.87rem;font-weight:600;background:#111827;color:#e6eef8;border:1px solid #334155}
.tab.active{background:#00d4ff;color:#000;border-color:#00d4ff}
.tab-content{display:none}.tab-content.active{display:block}
.progress-bar{background:#0d0f1a;border-radius:20px;overflow:hidden;height:10px;border:1px solid #2d3244}
.progress-fill{height:100%;background:linear-gradient(90deg,#00d4ff,#0078ff);transition:width .3s}
.thumb-wrap{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.thumb-wrap img{width:56px;height:56px;border-radius:5px;border:2px solid #00d4ff;object-fit:cover}
video{border-radius:8px;border:2px solid #2d3244;background:#000;width:300px;height:225px}
.checkbox-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-top:8px}
.checkbox-item{display:flex;align-items:center;gap:8px;background:#111827;padding:10px 14px;border-radius:7px;border:1px solid #2d3244;cursor:pointer}
.checkbox-item input{width:auto}
.checkbox-item:hover{border-color:#00d4ff}
.narrow{max-width:480px;margin:0 auto}
.link-row{text-align:center;margin-top:14px;font-size:.87rem;color:#aaa}
.export-bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:16px;padding:12px 16px;background:#111827;border-radius:8px;border:1px solid #1f2937}
.export-bar span{font-size:.82rem;color:#888}
</style>
"""

NAV_PUBLIC = """
<div class="topbar">
  <h1>Attendance System</h1>
  <nav>
    <a href="/student">Student Registration</a>
    <a href="/professor/register">Professor Registration</a>
    <a href="/professor/login">Professor Login</a>
    <a href="/hod/register_page">HOD Registration</a>
    <a href="/hod/login">HOD Login</a>
    <a href="/student/login">Student Login</a>
  </nav>
</div>
"""

NAV_PROFESSOR = """
<div class="topbar">
  <h1>Attendance System</h1>
  <nav>
    <a href="/professor/dashboard">Dashboard</a>
    <a href="/professor/mobile-attendance">Mobile Attendance</a>
    <a href="/professor/logout">Logout</a>
  </nav>
</div>
"""

NAV_HOD = """
<div class="topbar">
  <h1>Attendance System</h1>
  <nav>
    <a href="/hod/dashboard">Dashboard</a>
    <a href="/hod/logout">Logout</a>
  </nav>
</div>
"""

NAV_STUDENT = """
<div class="topbar">
  <h1>Attendance System</h1>
  <nav>
    <a href="/student/dashboard">My Attendance</a>
    <a href="/student/analytics">Analytics</a>
    <a href="/student/logout">Logout</a>
  </nav>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

STUDENT_REG_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card">
  <h2>Student Registration</h2>
  <div class="grid">
    <div><label>IEN</label><input id="ien" placeholder="2024COMPS001"></div>
    <div><label>Full Name</label><input id="sname" placeholder="Alice Sharma"></div>
    <div><label>Email</label><input id="email" type="email" placeholder="alice@college.edu"></div>
    <div><label>Mobile</label><input id="mobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="password" type="password" placeholder="Min 6 chars"></div>
    <div><label>Roll Number</label><input id="roll_no" placeholder="42"></div>
    <div><label>Branch</label>
      <select id="branch"><option value="">-- Branch --</option>
      {% for b in branches %}<option>{{b}}</option>{% endfor %}</select>
    </div>
    <div><label>Year</label>
      <select id="year"><option value="">-- Year --</option>
      {% for y in years %}<option>{{y}}</option>{% endfor %}</select>
    </div>
    <div><label>Division (optional)</label>
      <select id="div"><option value="">-- None --</option>
      {% for d in divs %}<option>{{d}}</option>{% endfor %}</select>
    </div>
  </div>
  <div style="margin-top:24px">
    <h3>Face Registration ({{samples}} photos required)</h3>
    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
      <div>
        <video id="video" autoplay playsinline></video><br>
        <button class="btn btn-primary" style="margin-top:8px;width:300px" onclick="capture()">📷 Capture Photo</button>
        <div class="progress-bar" style="margin-top:8px"><div class="progress-fill" id="prog" style="width:0%"></div></div>
        <div id="camStatus" style="margin-top:6px;font-size:.85rem;color:#aaa">0 / {{samples}} captured</div>
      </div>
      <div class="thumb-wrap" id="thumbs"></div>
    </div>
  </div>
  <button class="btn btn-success btn-full" id="subBtn" onclick="submitReg()" disabled>Register Student</button>
  <div id="msg" style="margin-top:10px;font-size:.9rem;min-height:20px"></div>
</div></div>
<script>
const SAMPLES={{samples}};let captured=[];
navigator.mediaDevices.getUserMedia({video:true}).then(s=>document.getElementById('video').srcObject=s).catch(()=>{});
async function capture(){
  if(captured.length>=SAMPLES) return;
  const v=document.getElementById('video'),c=document.createElement('canvas');
  c.width=v.videoWidth||320;c.height=v.videoHeight||240;c.getContext('2d').drawImage(v,0,0);
  const url=c.toDataURL('image/jpeg',.85);
  const status=document.getElementById('camStatus');
  status.style.color='#aaa';status.textContent='Checking for face...';
  const res=await fetch('/check_face',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:url})});
  const r=await res.json();
  if(!r.found){status.style.color='#ff5252';status.textContent='No face detected. Try again.';return;}
  captured.push(url);
  const img=document.createElement('img');img.src=url;document.getElementById('thumbs').appendChild(img);
  document.getElementById('prog').style.width=(captured.length/SAMPLES*100)+'%';
  status.style.color='#aaa';status.textContent=captured.length+' / '+SAMPLES+' captured';
  if(captured.length>=SAMPLES){document.getElementById('subBtn').disabled=false;
    status.style.color='#00e676';status.textContent='✓ All photos verified. Ready to register.';}
}
async function submitReg(){
  const data={ien:document.getElementById('ien').value,name:document.getElementById('sname').value,
    email:document.getElementById('email').value,mobile:document.getElementById('mobile').value,
    password:document.getElementById('password').value,roll_no:document.getElementById('roll_no').value,
    branch:document.getElementById('branch').value,year:document.getElementById('year').value,
    div:document.getElementById('div').value,images:captured};
  document.getElementById('subBtn').disabled=true;
  document.getElementById('msg').textContent='Registering...';
  const res=await fetch('/student/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const r=await res.json();
  const el=document.getElementById('msg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=(r.success?'✓ ':'')+r.message;
  if(!r.success)document.getElementById('subBtn').disabled=false;
}
</script>
"""

PROF_REG_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card narrow">
  <h2>Professor Registration</h2>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label>Full Name</label><input id="pname" placeholder="Dr. Ramesh Sharma"></div>
    <div><label>Mobile Number</label><input id="pmobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="ppassword" type="password" placeholder="Min 6 chars"></div>
    <div>
      <label>Branches (select all that apply)</label>
      <div class="checkbox-grid">
        {% for b in branches %}
        <label class="checkbox-item"><input type="checkbox" name="branch" value="{{b}}"> {{b}}</label>
        {% endfor %}
      </div>
    </div>
  </div>
  <div style="margin-top:24px">
    <h3>Face Registration ({{samples}} photos required)</h3>
    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
      <div>
        <video id="video" autoplay playsinline></video><br>
        <button class="btn btn-primary" style="margin-top:8px;width:300px" onclick="capture()">📷 Capture Photo</button>
        <div class="progress-bar" style="margin-top:8px"><div class="progress-fill" id="prog" style="width:0%"></div></div>
        <div id="camStatus" style="margin-top:6px;font-size:.85rem;color:#aaa">0 / {{samples}} captured</div>
      </div>
      <div class="thumb-wrap" id="thumbs"></div>
    </div>
  </div>
  <button class="btn btn-success btn-full" id="subBtn" onclick="submitProf()" disabled>Register Professor</button>
  <div id="msg" style="margin-top:10px;font-size:.9rem;min-height:20px;text-align:center"></div>
  <div class="link-row">Already registered? <a href="/professor/login">Login here</a></div>
</div></div>
<script>
const SAMPLES={{samples}};let captured=[];
navigator.mediaDevices.getUserMedia({video:true}).then(s=>document.getElementById('video').srcObject=s).catch(()=>{});
async function capture(){
  if(captured.length>=SAMPLES) return;
  const v=document.getElementById('video'),c=document.createElement('canvas');
  c.width=v.videoWidth||320;c.height=v.videoHeight||240;c.getContext('2d').drawImage(v,0,0);
  const url=c.toDataURL('image/jpeg',.85);
  const status=document.getElementById('camStatus');
  status.style.color='#aaa';status.textContent='Checking for face...';
  const res=await fetch('/check_face',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:url})});
  const r=await res.json();
  if(!r.found){status.style.color='#ff5252';status.textContent='No face detected. Try again.';return;}
  captured.push(url);
  const img=document.createElement('img');img.src=url;document.getElementById('thumbs').appendChild(img);
  document.getElementById('prog').style.width=(captured.length/SAMPLES*100)+'%';
  status.style.color='#aaa';status.textContent=captured.length+' / '+SAMPLES+' captured';
  if(captured.length>=SAMPLES){document.getElementById('subBtn').disabled=false;
    status.style.color='#00e676';status.textContent='✓ All photos verified. Ready to register.';}
}
async function submitProf(){
  const branches=[...document.querySelectorAll('input[name=branch]:checked')].map(e=>e.value);
  const data={name:document.getElementById('pname').value,mobile:document.getElementById('pmobile').value,
    password:document.getElementById('ppassword').value,branches,images:captured};
  document.getElementById('subBtn').disabled=true;
  document.getElementById('msg').textContent='Registering...';
  const res=await fetch('/professor/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const r=await res.json();
  const el=document.getElementById('msg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=(r.success?'✓ ':'')+r.message;
  if(r.success)setTimeout(()=>window.location='/professor/login',1500);
  else document.getElementById('subBtn').disabled=false;
}
</script>
"""

PROF_LOGIN_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card narrow">
  <h2>Professor Login</h2>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label>Mobile Number</label><input id="lmobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="lpassword" type="password"></div>
    <button class="btn btn-primary btn-full" onclick="doLogin()">Login</button>
    <div id="loginMsg" style="text-align:center;font-size:.9rem;min-height:20px"></div>
  </div>
  <div class="link-row">New professor? <a href="/professor/register">Register here</a></div>
</div></div>
<script>
async function doLogin(){
  const res=await fetch('/professor/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mobile:document.getElementById('lmobile').value,password:document.getElementById('lpassword').value})});
  const r=await res.json();
  const el=document.getElementById('loginMsg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=r.message;
  if(r.success)setTimeout(()=>window.location='/professor/dashboard',800);
}
</script>
"""

# ─── Professor Dashboard ──────────────────────────────────────────────────────
# FIX: export link updates whenever branch filter changes; export buttons are
#       always visible and labelled clearly.
PROF_DASH_PAGE = BASE_STYLE + NAV_PROFESSOR + """
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
    <div>
      <h2 style="margin-bottom:6px">Professor Dashboard</h2>
      <span style="color:#aaa;font-size:.9rem">{{prof_name}}</span>
      {% for b in branches %}<span class="badge badge-branch" style="margin-left:6px">{{b}}</span>{% endfor %}
    </div>
  </div>

  <div class="stat-grid">
    <div class="stat-card"><div class="num">{{total_sessions}}</div><div class="lbl">Sessions Taken</div></div>
    <div class="stat-card"><div class="num">{{total_present}}</div><div class="lbl">Total Present</div></div>
    <div class="stat-card"><div class="num">{{total_absent}}</div><div class="lbl">Total Absent</div></div>
  </div>

  <div class="card">
    <h3>My Sessions</h3>

    <!-- Filter + Export bar (always visible) -->
    <div class="export-bar">
      <span>Filter:</span>
      <select id="branch-filter" onchange="filterSessions(this.value)" style="width:auto;min-width:160px">
        <option value="">All Branches</option>
        {% for b in branches %}<option>{{b}}</option>{% endfor %}
      </select>
      <span style="margin-left:auto"></span>
      <a id="export-all-btn" href="/professor/export" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">
        ⬇ Export All Sessions CSV
      </a>
      <a id="export-filtered-btn" href="/professor/export" class="btn btn-info" style="padding:7px 16px;font-size:.84rem;display:none">
        ⬇ Export Filtered CSV
      </a>
    </div>

    <div id="sessions-table">
    <table>
      <thead><tr><th>#</th><th>Branch</th><th>Year</th><th>Subject</th><th>Date & Time</th><th>Present</th><th>Absent</th><th>%</th><th>Detail</th></tr></thead>
      <tbody>
      {% for s in sessions %}
      {% set total=s.present+s.absent %}
      <tr>
        <td>{{s.id}}</td>
        <td><span class="badge badge-branch">{{s.branch}}</span></td>
        <td>{{s.year}}</td><td>{{s.subject}}</td>
        <td>{{s.start[:16].replace('T',' ')}}</td>
        <td><span class="badge badge-present">{{s.present}}</span></td>
        <td><span class="badge badge-absent">{{s.absent}}</span></td>
        <td>{{'%.1f'|format(s.present/total*100)+'%' if total>0 else '—'}}</td>
        <td><a href="/professor/session/{{s.id}}" style="color:#00d4ff">View →</a></td>
      </tr>
      {% else %}
      <tr><td colspan="9" style="text-align:center;color:#666;padding:24px">No sessions found.</td></tr>
      {% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>
<script>
let currentBranch = '';

async function filterSessions(branch){
  currentBranch = branch;
  const res = await fetch('/professor/sessions?branch=' + encodeURIComponent(branch));
  const rows = await res.json();

  let html = '<table><thead><tr><th>#</th><th>Branch</th><th>Year</th><th>Subject</th><th>Date & Time</th><th>Present</th><th>Absent</th><th>%</th><th>Detail</th></tr></thead><tbody>';
  if(rows.length === 0){
    html += '<tr><td colspan="9" style="text-align:center;color:#666;padding:24px">No sessions found.</td></tr>';
  } else {
    for(const s of rows){
      const total = s.present + s.absent;
      const pct = total > 0 ? (s.present/total*100).toFixed(1)+'%' : '—';
      html += `<tr>
        <td>${s.id}</td>
        <td><span class="badge badge-branch">${s.branch}</span></td>
        <td>${s.year}</td><td>${s.subject}</td>
        <td>${s.start.replace('T',' ').slice(0,16)}</td>
        <td><span class="badge badge-present">${s.present}</span></td>
        <td><span class="badge badge-absent">${s.absent}</span></td>
        <td>${pct}</td>
        <td><a href="/professor/session/${s.id}" style="color:#00d4ff">View →</a></td>
      </tr>`;
    }
  }
  html += '</tbody></table>';
  document.getElementById('sessions-table').innerHTML = html;

  // Update export buttons
  const allBtn = document.getElementById('export-all-btn');
  const filteredBtn = document.getElementById('export-filtered-btn');
  if(branch){
    allBtn.style.display = 'none';
    filteredBtn.style.display = 'inline-block';
    filteredBtn.href = '/professor/export?branch=' + encodeURIComponent(branch);
    filteredBtn.textContent = '⬇ Export ' + branch + ' CSV';
  } else {
    allBtn.style.display = 'inline-block';
    filteredBtn.style.display = 'none';
  }
}
</script>
"""

# ─── HOD Dashboard ────────────────────────────────────────────────────────────
# FIX: each tab has its own clearly labelled export button that updates with
#       the active filter. Export buttons always visible.
HOD_DASH = BASE_STYLE + NAV_HOD + """
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
    <div>
      <h2 style="margin-bottom:4px">HOD Dashboard</h2>
      <span class="badge badge-branch">{{branch}}</span>
      <span style="margin-left:10px;color:#aaa;font-size:.9rem">{{hod_name}}</span>
    </div>
    <a href="/hod/logout" class="btn btn-danger">Logout</a>
  </div>

  <div class="stat-grid">
    <div class="stat-card"><div class="num">{{total_students}}</div><div class="lbl">Students</div></div>
    <div class="stat-card"><div class="num">{{total_professors}}</div><div class="lbl">Professors</div></div>
    <div class="stat-card"><div class="num">{{total_sessions}}</div><div class="lbl">Sessions</div></div>
    <div class="stat-card"><div class="num">{{avg_attendance}}%</div><div class="lbl">Avg Attendance</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('sessions')">Sessions</div>
    <div class="tab" onclick="switchTab('students')">Students</div>
    <div class="tab" onclick="switchTab('professors')">Professors</div>
  </div>

  <!-- ── SESSIONS TAB ──────────────────────────────────────────────────────── -->
  <div class="tab-content active" id="tab-sessions">
    <div class="card">
      <h3>All Sessions — {{branch}}</h3>

      <!-- Filter + Export bar -->
      <div class="export-bar">
        <span>Filter by Professor:</span>
        <select id="prof-filter" onchange="filterSessions(this.value)" style="width:auto;min-width:180px">
          <option value="">All Professors</option>
          {% for pid,pname in professors %}<option value="{{pid}}">{{pname}}</option>{% endfor %}
        </select>
        <span style="margin-left:auto"></span>
        <a id="sessions-export-btn" href="/hod/export/sessions" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">
          ⬇ Export All Sessions CSV
        </a>
      </div>

      <div id="sessions-table">
      <table>
        <thead><tr><th>#</th><th>Professor</th><th>Year</th><th>Subject</th><th>Date & Time</th><th>Present</th><th>Absent</th><th>%</th></tr></thead>
        <tbody>
        {% for sid,pname,year,br,subject,start,present,absent in sessions %}
        {% set total=present+absent %}
        <tr>
          <td>{{sid}}</td><td>{{pname}}</td><td>{{year}}</td><td>{{subject}}</td>
          <td>{{start[:16].replace('T',' ')}}</td>
          <td><span class="badge badge-present">{{present}}</span></td>
          <td><span class="badge badge-absent">{{absent}}</span></td>
          <td>{{'%.1f'|format(present/total*100)+'%' if total>0 else '—'}}</td>
        </tr>
        {% else %}
        <tr><td colspan="8" style="text-align:center;color:#666;padding:24px">No sessions found.</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
  </div>

  <!-- ── STUDENTS TAB ──────────────────────────────────────────────────────── -->
  <div class="tab-content" id="tab-students">
    <div class="card">
      <h3>Students — {{branch}}</h3>

      <!-- Filter + Export bar -->
      <div class="export-bar">
        <span>Filter by Year:</span>
        <select id="year-filter" onchange="refreshStudents()" style="width:auto;min-width:140px">
          <option value="">All Years</option>
          {% for y in years %}<option>{{y}}</option>{% endfor %}
        </select>
        <span>Sort by:</span>
        <select id="student-sort-by" onchange="refreshStudents()" style="width:auto;min-width:150px">
          <option value="name">Name</option>
          <option value="year">Year</option>
          <option value="roll">Roll</option>
          <option value="present">Present</option>
          <option value="absent">Absent</option>
        </select>
        <select id="student-sort-order" onchange="refreshStudents()" style="width:auto;min-width:120px">
          <option value="asc">Ascending</option>
          <option value="desc">Descending</option>
        </select>
        <span style="margin-left:auto"></span>
        <a id="students-export-btn" href="/hod/export/students" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">
          ⬇ Export All Students CSV
        </a>
      </div>

      <div id="students-table">
      <table>
        <thead><tr><th>Name</th><th>Year</th><th>Div</th><th>Roll</th><th>Present</th><th>Absent</th><th>Attendance %</th><th>Actions</th></tr></thead>
        <tbody>
        {% for name,year,div,roll,present,absent,student_id in student_summary %}
        {% set total=present+absent %}
        {% set pct=((present/total)*100)|round(1) if total>0 else 0 %}
        <tr>
          <td><a href="/hod/student/{{student_id}}/attendance" style="color:#00d4ff">{{name}}</a></td>
          <td>{{year}}</td><td>{{div or '—'}}</td><td>{{roll}}</td>
          <td><span class="badge badge-present">{{present}}</span></td>
          <td><span class="badge badge-absent">{{absent}}</span></td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div class="progress-bar" style="width:80px;height:8px">
                <div class="progress-fill" style="width:{{pct}}%;background:{{'#00c853' if pct>=75 else '#ff9800' if pct>=50 else '#ff5252'}}"></div>
              </div>
              <span style="font-size:.82rem;color:{{'#00e676' if pct>=75 else '#ffb74d' if pct>=50 else '#ff5252'}}">{{pct}}%</span>
            </div>
          </td>
          <td>
            <a href="/hod/student/{{student_id}}/attendance" class="btn btn-info" style="padding:5px 10px;font-size:.78rem;margin-right:4px">Detail</a>
            <button class="btn btn-primary" style="padding:5px 10px;font-size:.78rem;margin-right:4px" onclick="openStudentProfile({{student_id}})">Profile</button>
            <button class="btn btn-danger" style="padding:5px 10px;font-size:.78rem" onclick="deleteStudent({{student_id}},'{{name}}')">Delete</button>
          </td>
        </tr>
        {% else %}
        <tr><td colspan="8" style="text-align:center;color:#666;padding:24px">No students found.</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
  </div>

  <!-- ── PROFESSORS TAB ────────────────────────────────────────────────────── -->
  <div class="tab-content" id="tab-professors">
    <div class="card">
      <h3>Professors teaching {{branch}}</h3>

      <!-- Export bar -->
      <div class="export-bar">
        <span>Sort by:</span>
        <select id="prof-sort-by" onchange="refreshProfessors()" style="width:auto;min-width:150px">
          <option value="name">Name</option>
          <option value="sessions">Sessions</option>
          <option value="mobile">Mobile</option>
        </select>
        <select id="prof-sort-order" onchange="refreshProfessors()" style="width:auto;min-width:120px">
          <option value="asc">Ascending</option>
          <option value="desc">Descending</option>
        </select>
        <span style="margin-left:auto"></span>
        <a href="/hod/export/professors" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">
          ⬇ Export Professors CSV
        </a>
      </div>

      <div id="professors-table">
      <table>
        <thead><tr><th>#</th><th>Name</th><th>Sessions</th><th>Detail</th></tr></thead>
        <tbody>
        {% for pid,pname in professors %}
        <tr>
          <td>{{pid}}</td><td>{{pname}}</td>
          <td>{{prof_session_counts.get(pid,0)}}</td>
          <td>
            <button class="btn btn-primary" style="padding:5px 10px;font-size:.78rem;margin-right:4px" onclick="openProfessorProfile({{pid}})">Profile</button>
            <a href="/hod/professor/{{pid}}">View →</a>
          </td>
        </tr>
        {% else %}
        <tr><td colspan="4" style="text-align:center;color:#666;padding:24px">No professors found.</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
  </div>

  <div id="profile-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:1000;padding:20px;overflow:auto">
    <div style="max-width:760px;margin:20px auto;background:#111827;border:1px solid #334155;border-radius:12px;padding:22px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;gap:10px;flex-wrap:wrap">
        <h3 id="profile-title" style="margin:0;color:#00d4ff">Profile</h3>
        <button class="btn btn-danger" style="padding:6px 12px" onclick="hideProfileModal()">Close</button>
      </div>
      <div id="profile-body" style="font-size:.92rem;color:#e6eef8"></div>
    </div>
  </div>

</div>
<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name){
  const ids = ['sessions','students','professors'];
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ids[i] === name);
    document.getElementById('tab-' + ids[i]).classList.toggle('active', ids[i] === name);
  });
}

// ── Sessions filter & export update ──────────────────────────────────────────
async function filterSessions(pid){
  const res = await fetch('/hod/sessions?professor_id=' + (pid || ''));
  const r   = await res.json();
  let html  = '<table><thead><tr><th>#</th><th>Professor</th><th>Year</th><th>Subject</th><th>Date & Time</th><th>Present</th><th>Absent</th><th>%</th></tr></thead><tbody>';
  if(r.sessions.length === 0){
    html += '<tr><td colspan="8" style="text-align:center;color:#666;padding:24px">No sessions found.</td></tr>';
  } else {
    for(const s of r.sessions){
      const total = s.present + s.absent;
      const pct   = total > 0 ? (s.present/total*100).toFixed(1)+'%' : '—';
      html += `<tr><td>${s.id}</td><td>${s.professor}</td><td>${s.year}</td><td>${s.subject}</td>
               <td>${s.start.replace('T',' ').slice(0,16)}</td>
               <td><span class="badge badge-present">${s.present}</span></td>
               <td><span class="badge badge-absent">${s.absent}</span></td>
               <td>${pct}</td></tr>`;
    }
  }
  html += '</tbody></table>';
  document.getElementById('sessions-table').innerHTML = html;

  // Update export button label + URL
  const btn = document.getElementById('sessions-export-btn');
  if(pid){
    const sel  = document.getElementById('prof-filter');
    const name = sel.options[sel.selectedIndex].text;
    btn.href   = '/hod/export/sessions?professor_id=' + encodeURIComponent(pid);
    btn.textContent = '⬇ Export ' + name + ' Sessions CSV';
  } else {
    btn.href = '/hod/export/sessions';
    btn.textContent = '⬇ Export All Sessions CSV';
  }
}

// ── Students filter & export update ──────────────────────────────────────────
async function filterStudents(year){
  if (typeof year === 'string') {
    const yearEl = document.getElementById('year-filter');
    if (yearEl) yearEl.value = year;
  }
  await refreshStudents();
}

function escapeHtml(value){
  return String(value ?? '—')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showProfileModal(title, bodyHtml){
  document.getElementById('profile-title').textContent = title;
  document.getElementById('profile-body').innerHTML = bodyHtml;
  document.getElementById('profile-modal').style.display = 'block';
}

function hideProfileModal(){
  document.getElementById('profile-modal').style.display = 'none';
}

async function openStudentProfile(studentId){
  const res = await fetch('/hod/student/' + studentId + '/profile');
  const data = await res.json();
  if(!data.success){
    alert('✗ Error: ' + (data.message || 'Unable to load profile'));
    return;
  }
  const p = data.profile;
  showProfileModal(
    'Student Profile',
    `<table>
      <tbody>
        <tr><th style="width:180px">Name</th><td>${escapeHtml(p.name)}</td></tr>
        <tr><th>IEN</th><td>${escapeHtml(p.ien)}</td></tr>
        <tr><th>Email</th><td>${escapeHtml(p.email)}</td></tr>
        <tr><th>Mobile</th><td>${escapeHtml(p.mobile)}</td></tr>
        <tr><th>Branch</th><td>${escapeHtml(p.branch)}</td></tr>
        <tr><th>Year</th><td>${escapeHtml(p.year)}</td></tr>
        <tr><th>Division</th><td>${escapeHtml(p.div || '—')}</td></tr>
        <tr><th>Roll</th><td>${escapeHtml(p.roll_no)}</td></tr>
        <tr><th>Present</th><td>${escapeHtml(p.present)}</td></tr>
        <tr><th>Absent</th><td>${escapeHtml(p.absent)}</td></tr>
      </tbody>
    </table>`
  );
}

async function openProfessorProfile(professorId){
  const res = await fetch('/hod/professor/' + professorId + '/profile');
  const data = await res.json();
  if(!data.success){
    alert('✗ Error: ' + (data.message || 'Unable to load profile'));
    return;
  }
  const p = data.profile;
  showProfileModal(
    'Professor Profile',
    `<table>
      <tbody>
        <tr><th style="width:180px">Name</th><td>${escapeHtml(p.name)}</td></tr>
        <tr><th>Mobile</th><td>${escapeHtml(p.mobile)}</td></tr>
        <tr><th>Assigned Branches</th><td>${escapeHtml((p.branches || []).join(', '))}</td></tr>
        <tr><th>Sessions (This Branch)</th><td>${escapeHtml(p.sessions)}</td></tr>
        <tr><th>Total Present</th><td>${escapeHtml(p.present)}</td></tr>
        <tr><th>Total Absent</th><td>${escapeHtml(p.absent)}</td></tr>
      </tbody>
    </table>`
  );
}

async function refreshStudents(){
  const year = document.getElementById('year-filter')?.value || '';
  const sortBy = document.getElementById('student-sort-by')?.value || 'name';
  const sortOrder = document.getElementById('student-sort-order')?.value || 'asc';
  const query = new URLSearchParams({year, sort_by: sortBy, sort_order: sortOrder});
  const res = await fetch('/hod/students?' + query.toString());
  const r   = await res.json();
  let html  = '<table><thead><tr><th>Name</th><th>Year</th><th>Div</th><th>Roll</th><th>Present</th><th>Absent</th><th>Attendance %</th><th>Actions</th></tr></thead><tbody>';
  if(r.students.length === 0){
    html += '<tr><td colspan="8" style="text-align:center;color:#666;padding:24px">No students found.</td></tr>';
  } else {
    for(const s of r.students){
      const total = s.present + s.absent;
      const pct   = total > 0 ? ((s.present/total)*100).toFixed(1) : 0;
      const color = pct >= 75 ? '#00c853' : pct >= 50 ? '#ff9800' : '#ff5252';
      const tcolor= pct >= 75 ? '#00e676' : pct >= 50 ? '#ffb74d' : '#ff5252';
      html += `<tr>
        <td><a href="/hod/student/${s.id}/attendance" style="color:#00d4ff">${s.name}</a></td>
        <td>${s.year}</td><td>${s.div||'—'}</td><td>${s.roll}</td>
        <td><span class="badge badge-present">${s.present}</span></td>
        <td><span class="badge badge-absent">${s.absent}</span></td>
        <td><div style="display:flex;align-items:center;gap:8px">
          <div class="progress-bar" style="width:80px;height:8px">
            <div class="progress-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <span style="font-size:.82rem;color:${tcolor}">${pct}%</span>
        </div></td>
        <td>
          <a href="/hod/student/${s.id}/attendance" class="btn btn-info" style="padding:5px 10px;font-size:.78rem;margin-right:4px">Detail</a>
          <button class="btn btn-primary" style="padding:5px 10px;font-size:.78rem;margin-right:4px" onclick="openStudentProfile(${s.id})">Profile</button>
          <button class="btn btn-danger" style="padding:5px 10px;font-size:.78rem" onclick="deleteStudent(${s.id},'${s.name}')">Delete</button>
        </td>
      </tr>`;
    }
  }
  html += '</tbody></table>';
  document.getElementById('students-table').innerHTML = html;

  // Update export button
  const btn = document.getElementById('students-export-btn');
  if(year){
    btn.href = '/hod/export/students?year=' + encodeURIComponent(year);
    btn.textContent = '⬇ Export ' + year + ' Students CSV';
  } else {
    btn.href = '/hod/export/students';
    btn.textContent = '⬇ Export All Students CSV';
  }
}

async function refreshProfessors(){
  const sortBy = document.getElementById('prof-sort-by')?.value || 'name';
  const sortOrder = document.getElementById('prof-sort-order')?.value || 'asc';
  const query = new URLSearchParams({sort_by: sortBy, sort_order: sortOrder});
  const res = await fetch('/hod/professors?' + query.toString());
  const r = await res.json();
  let html = '<table><thead><tr><th>#</th><th>Name</th><th>Sessions</th><th>Detail</th></tr></thead><tbody>';
  if(r.professors.length === 0){
    html += '<tr><td colspan="4" style="text-align:center;color:#666;padding:24px">No professors found.</td></tr>';
  } else {
    for(const p of r.professors){
      html += `<tr>
        <td>${p.id}</td>
        <td>${escapeHtml(p.name)}</td>
        <td>${p.sessions}</td>
        <td>
          <button class="btn btn-primary" style="padding:5px 10px;font-size:.78rem;margin-right:4px" onclick="openProfessorProfile(${p.id})">Profile</button>
          <a href="/hod/professor/${p.id}">View →</a>
        </td>
      </tr>`;
    }
  }
  html += '</tbody></table>';
  document.getElementById('professors-table').innerHTML = html;
}

// ── Delete student ────────────────────────────────────────────────────────────
async function deleteStudent(studentId, studentName){
  if(!confirm('Delete ' + studentName + '? This cannot be undone.')) return;
  const res = await fetch('/hod/student/' + studentId + '/delete', {method:'POST'});
  const r   = await res.json();
  if(r.success){ alert('✓ ' + studentName + ' deleted.'); location.reload(); }
  else { alert('✗ Error: ' + r.message); }
}
</script>
"""

HOD_REG_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card narrow">
  <h2>HOD Registration</h2>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label>Full Name</label><input id="hname" placeholder="Dr. Suresh Mehta"></div>
    <div><label>Mobile Number</label><input id="hmobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="hpassword" type="password" placeholder="Min 6 chars"></div>
    <div><label>Branch (1 HOD per branch)</label>
      <select id="hbranch"><option value="">-- Select Branch --</option>
      {% for b in branches %}<option>{{b}}</option>{% endfor %}</select>
    </div>
    <button class="btn btn-success btn-full" onclick="regHod()">Register HOD</button>
    <div id="hodMsg" style="font-size:.9rem;min-height:20px;text-align:center"></div>
  </div>
  <div class="link-row">Already registered? <a href="/hod/login">Login here</a></div>
</div></div>
<script>
async function regHod(){
  const data={name:document.getElementById('hname').value,mobile:document.getElementById('hmobile').value,
    password:document.getElementById('hpassword').value,branch:document.getElementById('hbranch').value};
  const res=await fetch('/hod/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const r=await res.json();
  const el=document.getElementById('hodMsg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=(r.success?'✓ ':'')+r.message;
  if(r.success)setTimeout(()=>window.location='/hod/login',1500);
}
</script>
"""

HOD_LOGIN_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card narrow">
  <h2>HOD Login</h2>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label>Mobile Number</label><input id="lmobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="lpassword" type="password"></div>
    <button class="btn btn-primary btn-full" onclick="doLogin()">Login</button>
    <div id="loginMsg" style="text-align:center;font-size:.9rem;min-height:20px"></div>
  </div>
  <div class="link-row">New HOD? <a href="/hod/register_page">Register here</a></div>
</div></div>
<script>
async function doLogin(){
  const res=await fetch('/hod/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mobile:document.getElementById('lmobile').value,password:document.getElementById('lpassword').value})});
  const r=await res.json();
  const el=document.getElementById('loginMsg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=r.message;
  if(r.success)setTimeout(()=>window.location='/hod/dashboard',800);
}
</script>
"""

STUDENT_LOGIN_PAGE = BASE_STYLE + NAV_PUBLIC + """
<div class="container"><div class="card narrow">
  <h2>Student Login</h2>
  <div style="display:flex;flex-direction:column;gap:14px">
    <div><label>Mobile Number</label><input id="lmobile" placeholder="9876543210"></div>
    <div><label>Password</label><input id="lpassword" type="password"></div>
    <button class="btn btn-primary btn-full" onclick="doLogin()">Login</button>
    <div id="loginMsg" style="text-align:center;font-size:.9rem;min-height:20px"></div>
  </div>
  <div class="link-row">New student? <a href="/student">Register here</a></div>
</div></div>
<script>
async function doLogin(){
  const res=await fetch('/student/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mobile:document.getElementById('lmobile').value,
      password:document.getElementById('lpassword').value})});
  const r=await res.json();
  const el=document.getElementById('loginMsg');
  el.style.color=r.success?'#00e676':'#ff5252';el.textContent=r.message;
  if(r.success)setTimeout(()=>window.location='/student/dashboard',800);
}
</script>
"""

STUDENT_DASH_PAGE = BASE_STYLE + NAV_STUDENT + """
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
    <div>
      <h2 style="margin-bottom:6px">My Attendance</h2>
      <span style="color:#aaa;font-size:.9rem">{{student_name}}</span>
      <span class="badge badge-branch" style="margin-left:8px">{{year}} {{branch}}</span>
      <span style="margin-left:6px;color:#888;font-size:.85rem">Roll: {{roll_no}} | IEN: {{ien}}</span>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <a href="/student/export" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">⬇ Export My CSV</a>
      <a href="/student/analytics" class="btn btn-info">Analytics</a>
      <a href="/student/logout" class="btn btn-danger">Logout</a>
    </div>
  </div>

  <div class="card" style="padding:16px 20px">
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
      <label style="margin:0;white-space:nowrap">Filter by Subject:</label>
      <select id="subjectFilter" onchange="changeSubject(this.value)" style="min-width:220px;width:auto">
        <option value="all">All Subjects</option>
        {% for s in subjects %}
        <option value="{{s}}" {{'selected' if s==selected_subject else ''}}>{{s}}</option>
        {% endfor %}
      </select>
      <a id="subject-export-btn" href="/student/export{% if selected_subject != 'all' %}?subject={{selected_subject}}{% endif %}"
         class="btn btn-success" style="padding:7px 16px;font-size:.84rem;margin-left:auto">
        ⬇ Export {% if selected_subject != 'all' %}{{selected_subject}} {% endif %}CSV
      </a>
    </div>
  </div>

  <div class="stat-grid">
    <div class="stat-card"><div class="num" style="color:#00e676">{{present}}</div><div class="lbl">Present</div></div>
    <div class="stat-card"><div class="num" style="color:#ff5252">{{absent}}</div><div class="lbl">Absent</div></div>
    <div class="stat-card"><div class="num">{{total}}</div><div class="lbl">Total Sessions</div></div>
    <div class="stat-card">
      <div class="num" style="color:{{'#00e676' if pct>=75 else '#ff9800' if pct>=50 else '#ff5252'}}">{{pct}}%</div>
      <div class="lbl">Attendance %</div>
    </div>
  </div>

  <div class="card" style="padding:20px 24px">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
      <span style="font-size:.88rem;color:#aaa">Overall Progress</span>
      <span style="font-size:.88rem;color:{{'#00e676' if pct>=75 else '#ff9800' if pct>=50 else '#ff5252'}}">
        {{pct}}% {{'✓ Good' if pct>=75 else '⚠ Warning' if pct>=50 else '✗ Low'}}
      </span>
    </div>
    <div class="progress-bar" style="height:16px">
      <div class="progress-fill" style="width:{{pct}}%;background:{{'#00c853' if pct>=75 else '#ff9800' if pct>=50 else '#ff5252'}}"></div>
    </div>
    {% if pct < 75 %}
    <div style="margin-top:10px;padding:10px 14px;background:#2e1a00;border:1px solid #ff9800;border-radius:7px;font-size:.87rem;color:#ffb74d">
      Your attendance is below 75%. You need <strong>{{needed}}</strong> more present sessions to reach 75%.
    </div>
    {% endif %}
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('breakdown')">Subject Breakdown</div>
    <div class="tab" onclick="switchTab('history')">Session History</div>
  </div>

  <div class="tab-content active" id="tab-breakdown">
    <div class="card">
      <h3>Subject-wise Attendance</h3>
      <table>
        <thead><tr><th>Subject</th><th>Present</th><th>Absent</th><th>Total</th><th>Attendance %</th></tr></thead>
        <tbody>
        {% for subj, sp, sa, st in subject_breakdown %}
        {% set spct = ((sp/st)*100)|round(1) if st>0 else 0 %}
        <tr>
          <td>{{subj}}</td>
          <td><span class="badge badge-present">{{sp}}</span></td>
          <td><span class="badge badge-absent">{{sa}}</span></td>
          <td>{{st}}</td>
          <td>
            <div style="display:flex;align-items:center;gap:10px">
              <div class="progress-bar" style="width:100px;height:9px">
                <div class="progress-fill" style="width:{{spct}}%;background:{{'#00c853' if spct>=75 else '#ff9800' if spct>=50 else '#ff5252'}}"></div>
              </div>
              <span style="font-size:.83rem;min-width:40px;color:{{'#00e676' if spct>=75 else '#ffb74d' if spct>=50 else '#ff5252'}}">{{spct}}%</span>
            </div>
          </td>
        </tr>
        {% else %}
        <tr><td colspan="5" style="text-align:center;color:#666;padding:24px">No attendance records found.</td></tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <div class="tab-content" id="tab-history">
    <div class="card">
      <h3>Session History {% if selected_subject != 'all' %}— {{selected_subject}}{% endif %}</h3>
      <table>
        <thead><tr><th>Date & Time</th><th>Subject</th><th>Professor</th><th>Status</th><th>Confidence</th></tr></thead>
        <tbody>
        {% for subj, branch, year, start, status, conf, prof in history %}
        <tr>
          <td>{{start[:16].replace('T',' ')}}</td>
          <td>{{subj}}</td><td>{{prof}}</td>
          <td><span class="badge {{'badge-present' if status=='present' else 'badge-absent'}}">{{'✓ Present' if status=='present' else 'Absent'}}</span></td>
          <td style="color:#aaa">{{('%.2f'|format(conf)) if conf and conf>0 else '—'}}</td>
        </tr>
        {% else %}
        <tr><td colspan="5" style="text-align:center;color:#666;padding:24px">No records found.</td></tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
<script>
function switchTab(name){
  const ids=['breakdown','history'];
  document.querySelectorAll('.tab').forEach((t,i)=>{
    t.classList.toggle('active',ids[i]===name);
    document.getElementById('tab-'+ids[i]).classList.toggle('active',ids[i]===name);
  });
}
function changeSubject(val){
  window.location='/student/dashboard?subject='+encodeURIComponent(val);
}
</script>
"""

MOBILE_ATTENDANCE_PAGE = BASE_STYLE + NAV_PROFESSOR + """
<style>
.mobile-wrap{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}
.panel{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:20px}
.panel h3{margin-bottom:14px}
.kv{display:grid;grid-template-columns:120px 1fr;gap:8px 12px}
.kv div{padding:4px 0}
.muted{color:#aaa;font-size:.92rem}
.camera-box video{width:100%;max-width:100%;height:auto;aspect-ratio:4/3;background:#000}
.recent-list{display:flex;flex-direction:column;gap:8px;max-height:320px;overflow:auto}
.recent-item{background:#0d0f1a;border:1px solid #2d3244;border-radius:8px;padding:10px 12px}
.recent-item.ok{border-color:#00c853}
.recent-item.err{border-color:#ff5252}
@media(max-width:860px){.mobile-wrap{grid-template-columns:1fr}}
</style>
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:20px">
    <div>
      <h2 style="margin-bottom:6px">Mobile Attendance</h2>
      <div class="muted">Browser-based face capture for cloud/mobile use</div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <a href="/professor/dashboard" class="btn btn-warn">Back to Dashboard</a>
      {% if active_session %}<button class="btn btn-danger" onclick="endSession()">End Session</button>{% endif %}
    </div>
  </div>

  {% if active_session %}
  <div class="card" style="margin-bottom:16px">
    <h3>Active Session</h3>
    <div class="kv">
      <div>Professor</div><div>{{ active_session.professor_name }}</div>
      <div>Branch</div><div>{{ active_session.branch }}</div>
      <div>Year</div><div>{{ active_session.year }}</div>
      <div>Subject</div><div>{{ active_session.subject }}</div>
      <div>Recognized</div><div>{{ active_session.present_count }} / {{ active_session.total_students }}</div>
    </div>
  </div>
  {% else %}
  <div class="card" style="margin-bottom:16px">
    <h3>Start Attendance Session</h3>
    <div class="grid">
      <div><label>Branch</label><select id="branchSelect"></select></div>
      <div><label>Year</label><select id="yearSelect" onchange="syncSubjects()"></select></div>
      <div class="full"><label>Subject</label><select id="subjectSelect"></select></div>
    </div>
    <button class="btn btn-success btn-full" onclick="startSession()">Start Session</button>
  </div>
  {% endif %}

  <div class="mobile-wrap">
    <div class="panel camera-box">
      <h3>Camera Scan</h3>
      <video id="video" autoplay playsinline></video>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
        <button class="btn btn-primary" onclick="startCamera()">Open Camera</button>
        <button class="btn btn-primary" onclick="scanFace()">Scan Face</button>
        <button class="btn btn-warn" onclick="stopCamera()">Stop Camera</button>
      </div>
      <div id="scanStatus" class="muted" style="margin-top:8px">Use the camera to capture a student face, then tap Scan Face.</div>
    </div>
    <div class="panel">
      <h3>Scan Results</h3>
      <div class="recent-list" id="recentList"></div>
    </div>
  </div>
</div>
<script>
const YEARS = {{ years|tojson }};
const SUBJECTS = {{ subjects_map|tojson }};
const ALLOWED_BRANCHES = {{ allowed_branches|tojson }};
const ACTIVE_SESSION = {{ active_session|tojson }};
let cameraStream = null;

function setMessage(text, ok){
  const el = document.getElementById('scanStatus');
  el.textContent = text;
  el.style.color = ok ? '#00e676' : '#ff5252';
}
function addRecent(title, detail, ok){
  const list = document.getElementById('recentList');
  const item = document.createElement('div');
  item.className = 'recent-item ' + (ok ? 'ok' : 'err');
  item.innerHTML = '<strong>' + title + '</strong><div style="color:#aaa;font-size:.9rem;margin-top:4px">' + detail + '</div>';
  list.prepend(item);
}
function syncStartForm(){
  const branchSelect = document.getElementById('branchSelect');
  const yearSelect = document.getElementById('yearSelect');
  if (!branchSelect || !yearSelect) return;
  branchSelect.innerHTML = '';
  for (const b of ALLOWED_BRANCHES){
    const o = document.createElement('option'); o.value = b; o.textContent = b; branchSelect.appendChild(o);
  }
  yearSelect.innerHTML = '';
  for (const y of YEARS){
    const o = document.createElement('option'); o.value = y; o.textContent = y; yearSelect.appendChild(o);
  }
  syncSubjects();
}
function syncSubjects(){
  const yearSelect = document.getElementById('yearSelect');
  const subjectSelect = document.getElementById('subjectSelect');
  if (!yearSelect || !subjectSelect) return;
  const subjects = SUBJECTS[yearSelect.value] || [];
  subjectSelect.innerHTML = '';
  for (const s of subjects){
    const o = document.createElement('option'); o.value = s; o.textContent = s; subjectSelect.appendChild(o);
  }
}
async function startSession(){
  const branch = document.getElementById('branchSelect').value;
  const year = document.getElementById('yearSelect').value;
  const subject = document.getElementById('subjectSelect').value;
  const res = await fetch('/professor/mobile-attendance/start', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({branch,year,subject})
  });
  const data = await res.json();
  if(data.success) location.reload();
  else setMessage(data.message || 'Unable to start session', false);
}
async function endSession(){
  const res = await fetch('/professor/mobile-attendance/end', {method:'POST'});
  const data = await res.json();
  if(data.success) location.reload();
  else setMessage(data.message || 'Unable to end session', false);
}
async function startCamera(){
  if(cameraStream) return;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
    document.getElementById('video').srcObject = cameraStream;
    setMessage('Camera ready.', true);
  } catch(error){
    cameraStream = null;
    setMessage('Camera access failed: ' + error.message, false);
  }
}
function stopCamera(){
  if(!cameraStream) return;
  for(const t of cameraStream.getTracks()) t.stop();
  cameraStream = null;
  document.getElementById('video').srcObject = null;
  setMessage('Camera stopped.', true);
}
async function scanFace(){
  if(!cameraStream){ await startCamera(); if(!cameraStream) return; }
  const video = document.getElementById('video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth||640; canvas.height = video.videoHeight||480;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = canvas.toDataURL('image/jpeg', 0.9);
  setMessage('Scanning face...', true);
  const res = await fetch('/professor/mobile-attendance/scan', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({image})
  });
  const data = await res.json();
  if(data.success){ setMessage(data.message, true); addRecent(data.name, data.detail, true); }
  else { setMessage(data.message||'Scan failed', false); addRecent('Scan rejected', data.message||'No match', false); }
}
if(document.getElementById('branchSelect')) syncStartForm();
if(ACTIVE_SESSION) addRecent('Session active', ACTIVE_SESSION.branch+' '+ACTIVE_SESSION.year+' | '+ACTIVE_SESSION.subject, true);
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_face_embedding(b64_str):
    _, data = b64_str.split(',', 1)
    arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None
    encs = face_recognition.face_encodings(rgb, locs)
    return encs[0] if encs else None


def get_professor_by_mobile(mobile, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name FROM professors WHERE mobile=? AND password=?", (mobile, password))
    row = c.fetchone()
    conn.close()
    return row


def get_professor_sessions(professor_id, branch=None):
    conn = get_connection()
    c = conn.cursor()
    if branch:
        c.execute("""
            SELECT s.id, s.branch, s.year, s.subject, s.start_time,
                   COUNT(a.id) FILTER(WHERE a.status='present') as present,
                   COUNT(a.id) FILTER(WHERE a.status='absent')  as absent
            FROM sessions s LEFT JOIN attendance a ON a.session_id=s.id
            WHERE s.professor_id=? AND s.branch=?
            GROUP BY s.id ORDER BY s.start_time DESC
        """, (professor_id, branch))
    else:
        c.execute("""
            SELECT s.id, s.branch, s.year, s.subject, s.start_time,
                   COUNT(a.id) FILTER(WHERE a.status='present') as present,
                   COUNT(a.id) FILTER(WHERE a.status='absent')  as absent
            FROM sessions s LEFT JOIN attendance a ON a.session_id=s.id
            WHERE s.professor_id=?
            GROUP BY s.id ORDER BY s.start_time DESC
        """, (professor_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id":r[0],"branch":r[1],"year":r[2],"subject":r[3],
             "start":r[4],"present":r[5],"absent":r[6]} for r in rows]


def get_attendance_status(session_id, student_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT status FROM attendance WHERE session_id=? AND student_id=? ORDER BY id DESC LIMIT 1",
              (session_id, student_id))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def csv_response(filename, headers, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    response = Response(buffer.getvalue(), mimetype='text/csv; charset=utf-8')
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — STUDENT
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('student_page'))

@app.route('/student')
def student_page():
    return render_template_string(STUDENT_REG_PAGE,
        branches=BRANCHES, years=YEARS, divs=DIVS, samples=REGISTRATION_SAMPLES)

@app.route('/student/register', methods=['POST'])
def student_register():
    d = request.get_json()
    for f in ['ien','name','password','email','mobile','branch','year','roll_no','images']:
        if not d.get(f): return jsonify(success=False, message=f"Missing: {f}")
    if len(d['password']) < 6: return jsonify(success=False, message="Password min 6 characters.")
    if len(d['images']) < REGISTRATION_SAMPLES: return jsonify(success=False, message=f"Need {REGISTRATION_SAMPLES} photos.")
    embeddings = [e for b64 in d['images'] if (e := extract_face_embedding(b64)) is not None]
    if len(embeddings) < 5: return jsonify(success=False, message=f"Only {len(embeddings)} faces detected. Use better lighting.")
    sid, err = register_student(d['ien'], d['name'], d['password'], d['email'],
                                d['mobile'], d['branch'], d['year'],
                                d.get('div',''), d['roll_no'], np.mean(embeddings, axis=0))
    if err: return jsonify(success=False, message=err)
    return jsonify(success=True, message=f"'{d['name']}' registered successfully!")

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'GET':
        return render_template_string(STUDENT_LOGIN_PAGE)
    d = request.get_json()
    row = get_student_by_credentials(d.get('mobile',''), d.get('password',''))
    if not row:
        return jsonify(success=False, message="Invalid mobile or password.")
    session['student_id']     = row[0]
    session['student_name']   = row[1]
    session['student_branch'] = row[2]
    session['student_year']   = row[3]
    session['student_div']    = row[4]
    session['student_roll']   = row[5]
    session['student_ien']    = row[6]
    return jsonify(success=True, message="Login successful!")

@app.route('/student/logout')
def student_logout():
    for k in ['student_id','student_name','student_branch','student_year',
              'student_div','student_roll','student_ien']:
        session.pop(k, None)
    return redirect(url_for('student_login'))

@app.route('/student/dashboard')
def student_dashboard():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))
    sid              = session['student_id']
    selected_subject = request.args.get('subject', 'all')
    subjects         = student_get_subjects(sid)
    summary          = student_get_attendance_summary(sid, selected_subject)
    subject_breakdown = student_get_subject_breakdown(sid)
    history          = student_get_session_history(sid, selected_subject)
    present = summary[0] or 0
    absent  = summary[1] or 0
    total   = summary[2] or 0
    pct     = round((present / total) * 100, 1) if total > 0 else 0
    needed  = 0
    if pct < 75 and total > 0:
        needed = max(0, int((0.75 * total - present) / 0.25) + 1)
    return render_template_string(STUDENT_DASH_PAGE,
        student_name=session['student_name'],
        branch=session['student_branch'],
        year=session['student_year'],
        roll_no=session['student_roll'],
        ien=session['student_ien'],
        subjects=subjects,
        selected_subject=selected_subject,
        present=present, absent=absent, total=total, pct=pct,
        needed=needed,
        subject_breakdown=subject_breakdown,
        history=history)

@app.route('/student/export')
def student_export():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))
    sid     = session['student_id']
    subject = request.args.get('subject', '')
    history = student_get_session_history(sid, subject or 'all')
    rows = []
    for subj, branch, year, start, status, conf, prof in history:
        rows.append([start[:16].replace('T',' '), subj, prof, status.upper(),
                     f"{conf:.2f}" if conf and conf > 0 else '—'])
    name = session['student_name'].replace(' ', '_')
    suffix = f"_{subject.replace(' ','_')}" if subject else '_all'
    return csv_response(
        f"student_{name}{suffix}_attendance.csv",
        ['Date & Time', 'Subject', 'Professor', 'Status', 'Confidence'],
        rows
    )

@app.route('/student/analytics')
def student_analytics():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))
    sid = session['student_id']
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT s.subject, a.timestamp, a.status
        FROM attendance a JOIN sessions s ON a.session_id=s.id
        WHERE a.student_id=? ORDER BY a.timestamp DESC
    """, (sid,))
    records = c.fetchall()
    c.execute("""
        SELECT s.subject,
               COUNT(CASE WHEN a.status='present' THEN 1 END) as present,
               COUNT(CASE WHEN a.status='absent'  THEN 1 END) as absent
        FROM sessions s
        LEFT JOIN attendance a ON s.id=a.session_id AND a.student_id=?
        GROUP BY s.subject ORDER BY s.subject
    """, (sid,))
    subject_data = c.fetchall()
    conn.close()

    subjects = [r[0] for r in subject_data]
    present_counts = [r[1] or 0 for r in subject_data]
    absent_counts  = [r[2] or 0 for r in subject_data]

    month_present, month_absent = {}, {}
    for _, timestamp, status in records:
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            m = dt.strftime('%b %Y')
            month_present.setdefault(m, 0)
            month_absent.setdefault(m, 0)
            if status == 'present': month_present[m] += 1
            else: month_absent[m] += 1
    months = sorted(month_present.keys())
    monthly_present = [month_present.get(m,0) for m in months]
    monthly_absent  = [month_absent.get(m,0)  for m in months]
    total_present = sum(present_counts)
    total_absent  = sum(absent_counts)
    total         = total_present + total_absent
    overall_pct   = round((total_present/total)*100, 1) if total > 0 else 0
    overall_color = '#00c853' if overall_pct >= 75 else '#ff9800' if overall_pct >= 50 else '#ff5252'

    table_rows = ""
    for subj, present, absent in subject_data:
        t = (present or 0) + (absent or 0)
        p = round(((present or 0)/t)*100, 1) if t > 0 else 0
        c2 = '#00c853' if p >= 75 else '#ff9800' if p >= 50 else '#ff5252'
        table_rows += f"""<tr>
            <td>{subj}</td>
            <td><span class="badge badge-present">{present or 0}</span></td>
            <td><span class="badge badge-absent">{absent or 0}</span></td>
            <td>{t}</td>
            <td><div style="display:flex;align-items:center;gap:8px">
              <div class="progress-bar" style="width:80px;height:8px">
                <div class="progress-fill" style="width:{p}%;background:{c2}"></div>
              </div>
              <span style="font-size:.82rem">{p}%</span>
            </div></td>
        </tr>"""

    page = BASE_STYLE + NAV_STUDENT + f"""
    <div class="container">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <h2>Analytics & Performance</h2>
        <div style="display:flex;gap:10px">
          <a href="/student/export" class="btn btn-success" style="padding:7px 16px;font-size:.84rem">⬇ Export CSV</a>
          <a href="/student/dashboard" class="btn btn-primary">← Back</a>
        </div>
      </div>
      <div class="stat-grid">
        <div class="stat-card"><div class="num" style="color:#00e676">{total_present}</div><div class="lbl">Total Present</div></div>
        <div class="stat-card"><div class="num" style="color:#ff5252">{total_absent}</div><div class="lbl">Total Absent</div></div>
        <div class="stat-card"><div class="num">{total}</div><div class="lbl">Total Sessions</div></div>
        <div class="stat-card"><div class="num" style="color:{overall_color}">{overall_pct}%</div><div class="lbl">Overall</div></div>
      </div>
      <div class="card"><h3 style="margin-bottom:16px">Subject-wise Attendance</h3>
        <canvas id="subjectChart" style="max-height:300px"></canvas></div>
      <div class="card"><h3 style="margin-bottom:16px">Monthly Trend</h3>
        <canvas id="trendChart" style="max-height:300px"></canvas></div>
      <div class="card"><h3 style="margin-bottom:16px">Performance Metrics</h3>
        <table><thead><tr><th>Subject</th><th>Present</th><th>Absent</th><th>Total</th><th>Percentage</th></tr></thead>
        <tbody>{table_rows or '<tr><td colspan="5" style="text-align:center;color:#666;padding:24px">No data.</td></tr>'}</tbody></table>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script>
    new Chart(document.getElementById('subjectChart').getContext('2d'),{{
      type:'bar',
      data:{{labels:{str(subjects).replace("'",'"')},datasets:[
        {{label:'Present',data:{present_counts},backgroundColor:'#00c853',borderColor:'#00e676',borderWidth:1}},
        {{label:'Absent',data:{absent_counts},backgroundColor:'#ff5252',borderColor:'#ff6f6f',borderWidth:1}}
      ]}},
      options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#aaa'}}}}}},
        scales:{{y:{{ticks:{{color:'#aaa'}},grid:{{color:'#2d3244'}}}},x:{{ticks:{{color:'#aaa'}},grid:{{color:'#2d3244'}}}}}}}}
    }});
    new Chart(document.getElementById('trendChart').getContext('2d'),{{
      type:'line',
      data:{{labels:{str(months).replace("'",'"')},datasets:[
        {{label:'Present',data:{monthly_present},borderColor:'#00e676',backgroundColor:'rgba(0,230,118,0.1)',borderWidth:2,fill:true,tension:0.4,pointBackgroundColor:'#00e676',pointRadius:5}},
        {{label:'Absent',data:{monthly_absent},borderColor:'#ff5252',backgroundColor:'rgba(255,82,82,0.1)',borderWidth:2,fill:true,tension:0.4,pointBackgroundColor:'#ff5252',pointRadius:5}}
      ]}},
      options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#aaa'}}}}}},
        scales:{{y:{{ticks:{{color:'#aaa'}},grid:{{color:'#2d3244'}}}},x:{{ticks:{{color:'#aaa'}},grid:{{color:'#2d3244'}}}}}}}}
    }});
    </script>"""
    return page


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — PROFESSOR
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/professor/register')
def professor_register_page():
    return render_template_string(PROF_REG_PAGE, branches=BRANCHES, samples=REGISTRATION_SAMPLES)

@app.route('/professor/register', methods=['POST'])
def professor_register():
    d = request.get_json()
    for f in ['name','mobile','password','branches','images']:
        if not d.get(f): return jsonify(success=False, message=f"Missing: {f}")
    if len(d['password']) < 6: return jsonify(success=False, message="Password min 6 characters.")
    if not d['branches']: return jsonify(success=False, message="Select at least one branch.")
    if len(d['images']) < REGISTRATION_SAMPLES: return jsonify(success=False, message=f"Need {REGISTRATION_SAMPLES} photos.")
    embeddings = [e for b64 in d['images'] if (e := extract_face_embedding(b64)) is not None]
    if len(embeddings) < 5: return jsonify(success=False, message=f"Only {len(embeddings)} faces detected.")
    pid, err = register_professor(d['name'], d['mobile'], d['password'],
                                  d['branches'], np.mean(embeddings, axis=0))
    if err: return jsonify(success=False, message=err)
    return jsonify(success=True, message=f"Professor '{d['name']}' registered! (ID: {pid})")

@app.route('/professor/login', methods=['GET', 'POST'])
def professor_login():
    if request.method == 'GET':
        return render_template_string(PROF_LOGIN_PAGE)
    d = request.get_json()
    row = get_professor_by_mobile(d.get('mobile',''), d.get('password',''))
    if not row: return jsonify(success=False, message="Invalid mobile or password.")
    session['prof_id']   = row[0]
    session['prof_name'] = row[1]
    return jsonify(success=True, message="Login successful!")

@app.route('/professor/logout')
def professor_logout():
    session.pop('prof_id', None)
    session.pop('prof_name', None)
    return redirect(url_for('professor_login'))

@app.route('/professor/dashboard')
def professor_dashboard():
    if 'prof_id' not in session: return redirect(url_for('professor_login'))
    pid = session['prof_id']
    branches = get_professor_branches(pid)
    sessions_list = get_professor_sessions(pid)
    total_present = sum(s['present'] for s in sessions_list)
    total_absent  = sum(s['absent']  for s in sessions_list)
    return render_template_string(PROF_DASH_PAGE,
        prof_name=session['prof_name'], branches=branches,
        sessions=sessions_list, total_sessions=len(sessions_list),
        total_present=total_present, total_absent=total_absent)

@app.route('/professor/sessions')
def professor_sessions_api():
    if 'prof_id' not in session: return jsonify([])
    branch = request.args.get('branch','')
    return jsonify(get_professor_sessions(session['prof_id'], branch or None))

@app.route('/professor/export')
def professor_export():
    if 'prof_id' not in session: return redirect(url_for('professor_login'))
    branch = request.args.get('branch','')
    sessions_list = get_professor_sessions(session['prof_id'], branch or None)
    rows = []
    for s in sessions_list:
        total = s['present'] + s['absent']
        pct = round((s['present'] / total) * 100, 1) if total > 0 else 0.0
        rows.append([s['id'], s['branch'], s['year'], s['subject'],
                     s['start'].replace('T',' ')[:16], s['present'], s['absent'], pct])
    label = branch or 'all'
    return csv_response(
        f"professor_sessions_{label}.csv",
        ['Session ID','Branch','Year','Subject','Date & Time','Present','Absent','Attendance %'],
        rows
    )

# ── FIX: mobile routes are NOW at module level (not indented inside another fn) ──

@app.route('/professor/mobile-attendance')
def professor_mobile_attendance():
    if 'prof_id' not in session:
        return redirect(url_for('professor_login'))
    active_session = None
    mobile_session_id = session.get('mobile_session_id')
    if mobile_session_id:
        info = get_session_info(mobile_session_id)
        if info and info[5] is None:
            session_id, subject, branch, year, start_time, end_time, prof_name = info
            attendance = get_session_attendance(session_id)
            active_session = {
                'session_id': session_id, 'subject': subject,
                'branch': branch, 'year': year, 'start_time': start_time,
                'professor_name': prof_name,
                'present_count': sum(1 for r in attendance if r[5] == 'present'),
                'total_students': len(attendance),
            }
        else:
            for k in ['mobile_session_id','mobile_branch','mobile_year','mobile_subject']:
                session.pop(k, None)
    return render_template_string(MOBILE_ATTENDANCE_PAGE,
        prof_name=session['prof_name'],
        allowed_branches=get_professor_branches(session['prof_id']),
        years=YEARS, subjects_map=SUBJECTS, active_session=active_session)

@app.route('/professor/mobile-attendance/start', methods=['POST'])
def professor_mobile_attendance_start():
    if 'prof_id' not in session:
        return jsonify(success=False, message="Not logged in"), 401
    if session.get('mobile_session_id'):
        existing_info = get_session_info(session['mobile_session_id'])
        if existing_info and existing_info[5] is None:
            return jsonify(success=False, message="End the current mobile session first."), 400
        for k in ['mobile_session_id','mobile_branch','mobile_year','mobile_subject']:
            session.pop(k, None)
    d = request.get_json(silent=True) or {}
    branch  = d.get('branch','').strip()
    year    = d.get('year','').strip()
    subject = d.get('subject','').strip()
    if not branch or not year or not subject:
        return jsonify(success=False, message="Select branch, year, and subject."), 400
    if branch not in get_professor_branches(session['prof_id']):
        return jsonify(success=False, message="That branch is not assigned to you."), 403
    session_id    = create_session(session['prof_id'], branch, year, subject, datetime.now().isoformat())
    eligible_ids  = get_students_by_branch_year(branch, year)
    for student_id in eligible_ids:
        mark_attendance(session_id, student_id, datetime.now().isoformat(), 0.0, 'absent')
    session['mobile_session_id'] = session_id
    session['mobile_branch']  = branch
    session['mobile_year']    = year
    session['mobile_subject'] = subject
    return jsonify(success=True, message="Mobile attendance session started.", session_id=session_id)

@app.route('/professor/mobile-attendance/scan', methods=['POST'])
def professor_mobile_attendance_scan():
    if 'prof_id' not in session:
        return jsonify(success=False, message="Not logged in"), 401
    session_id   = session.get('mobile_session_id')
    session_info = get_session_info(session_id) if session_id else None
    if not session_id or not session_info or session_info[5] is not None:
        for k in ['mobile_session_id','mobile_branch','mobile_year','mobile_subject']:
            session.pop(k, None)
        return jsonify(success=False, message="Start a mobile attendance session first."), 400
    d     = request.get_json(silent=True) or {}
    image = d.get('image','')
    if not image:
        return jsonify(success=False, message="Missing camera image."), 400
    encoding = extract_face_embedding(image)
    if encoding is None:
        return jsonify(success=False, message="No face detected in the frame."), 200
    known = load_all_embeddings()
    pid, name, role, confidence = match_face(encoding, known)
    if pid is None or role != 'student':
        return jsonify(success=False, message="Student face not recognized."), 200
    existing_status = get_attendance_status(session_id, pid)
    if existing_status == 'present':
        return jsonify(success=False, message=f"{name} is already marked present."), 200
    if existing_status:
        update_attendance(session_id, pid, 'present')
    else:
        mark_attendance(session_id, pid, datetime.now().isoformat(), confidence, 'present')
    return jsonify(success=True, message=f"Marked present: {name}", name=name,
        detail=f"{session['mobile_branch']} {session['mobile_year']} | {session['mobile_subject']} | {confidence:.2f}")

@app.route('/professor/mobile-attendance/end', methods=['POST'])
def professor_mobile_attendance_end():
    if 'prof_id' not in session:
        return jsonify(success=False, message="Not logged in"), 401
    session_id   = session.get('mobile_session_id')
    branch       = session.get('mobile_branch')
    year         = session.get('mobile_year')
    session_info = get_session_info(session_id) if session_id else None
    if not session_id or not branch or not year or not session_info or session_info[5] is not None:
        for k in ['mobile_session_id','mobile_branch','mobile_year','mobile_subject']:
            session.pop(k, None)
        return jsonify(success=False, message="No active mobile session."), 400
    attendance   = get_session_attendance(session_id)
    present_ids  = {r[0] for r in attendance if r[5] == 'present'}
    eligible_ids = get_students_by_branch_year(branch, year)
    now = datetime.now().isoformat()
    mark_absent_students(session_id, present_ids, eligible_ids, now)
    close_session(session_id, now)
    for k in ['mobile_session_id','mobile_branch','mobile_year','mobile_subject']:
        session.pop(k, None)
    return jsonify(success=True, message="Mobile attendance session ended and saved.")

@app.route('/professor/session/<int:session_id>')
def professor_session_detail(session_id):
    if 'prof_id' not in session: return redirect(url_for('professor_login'))
    session_info = get_session_info(session_id)
    if not session_info: return redirect(url_for('professor_dashboard'))
    sid, subject, branch, year, start_time, end_time, prof_name = session_info
    attendance = get_session_attendance(session_id)
    present = [r for r in attendance if r[5] == 'present']
    absent  = [r for r in attendance if r[5] == 'absent']
    total   = len(present) + len(absent)
    pct     = round(len(present)/total*100, 1) if total > 0 else 0

    present_html = "".join([
        f"<tr><td>{r[2]}</td><td>{r[1]}</td><td>{r[4] or '—'}</td><td>{r[3]}</td>"
        f"<td><span style='color:#00e676'>{r[6]:.2f}</span></td>"
        f"<td><button class='btn btn-danger' style='padding:5px 10px;font-size:.8rem' onclick=\"removeAttendance({r[0]},'{r[1]}')\">Remove</button></td></tr>"
        for r in present])
    absent_html = "".join([
        f"<tr><td>{r[2]}</td><td>{r[1]}</td><td>{r[4] or '—'}</td><td>{r[3]}</td><td>—</td>"
        f"<td><button class='btn btn-success' style='padding:5px 10px;font-size:.8rem' onclick=\"markPresent({r[0]},'{r[1]}')\">Mark Present</button></td></tr>"
        for r in absent])

    page = BASE_STYLE + NAV_PROFESSOR + f"""
    <div class="container">
      <div style="margin-bottom:16px"><a href="/professor/dashboard">← Back to Dashboard</a></div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
          <div>
            <h2 style="margin-bottom:8px">{subject}</h2>
            <span style="color:#aaa;font-size:.9rem">{year} {branch} | {start_time[:16].replace('T',' ')}</span>
          </div>
          <a href="/professor/session/{session_id}/export" class="btn btn-success">⬇ Export CSV</a>
        </div>
        <div class="stat-grid">
          <div class="stat-card"><div class="num" style="color:#00e676">{len(present)}</div><div class="lbl">Present</div></div>
          <div class="stat-card"><div class="num" style="color:#ff5252">{len(absent)}</div><div class="lbl">Absent</div></div>
          <div class="stat-card"><div class="num">{total}</div><div class="lbl">Total</div></div>
          <div class="stat-card"><div class="num">{pct}%</div><div class="lbl">Attendance</div></div>
        </div>
        <h3 style="margin-top:24px;color:#00e676">✓ Present ({len(present)})</h3>
        <table style="margin-top:10px">
          <thead><tr><th>IEN</th><th>Name</th><th>Div</th><th>Roll</th><th>Confidence</th><th>Action</th></tr></thead>
          <tbody>{present_html or '<tr><td colspan="6" style="text-align:center;color:#666">No students present</td></tr>'}</tbody>
        </table>
        <h3 style="margin-top:28px;color:#ff5252">✗ Absent ({len(absent)})</h3>
        <table style="margin-top:10px">
          <thead><tr><th>IEN</th><th>Name</th><th>Div</th><th>Roll</th><th>Confidence</th><th>Action</th></tr></thead>
          <tbody>{absent_html or '<tr><td colspan="6" style="text-align:center;color:#666">No students absent</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <script>
    async function markPresent(studentId, studentName){{
      const res=await fetch('/professor/session/{session_id}/manual-mark',{{method:'POST',
        headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{student_id:studentId,status:'present'}})}});
      const r=await res.json();
      if(r.success){{alert('✓ '+studentName+' marked present');location.reload();}}
      else alert('✗ Error: '+r.message);
    }}
    async function removeAttendance(studentId, studentName){{
      if(!confirm('Remove '+studentName+' from attendance?')) return;
      const res=await fetch('/professor/session/{session_id}/manual-remove',{{method:'POST',
        headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{student_id:studentId}})}});
      const r=await res.json();
      if(r.success){{alert('✓ Attendance removed');location.reload();}}
      else alert('✗ Error: '+r.message);
    }}
    </script>"""
    return page

@app.route('/professor/session/<int:session_id>/export')
def professor_session_export(session_id):
    if 'prof_id' not in session: return redirect(url_for('professor_login'))
    session_info = get_session_info(session_id)
    if not session_info: return jsonify(error="Session not found"), 404
    sid, subject, branch, year, start_time, end_time, prof_name = session_info
    attendance = get_session_attendance(session_id)
    rows = []
    for stu_id, name, ien, roll, div, status, confidence, ts in attendance:
        rows.append([ien, name, div or '—', roll, status.upper(),
                     f"{confidence:.2f}" if confidence and confidence >= 0 else '—'])
    return csv_response(
        f"{branch}_{year}_{subject.replace(' ','_')}_attendance.csv",
        ['IEN','Student Name','Division','Roll No','Status','Confidence'], rows)

@app.route('/professor/session/<int:session_id>/manual-mark', methods=['POST'])
def professor_manual_mark(session_id):
    if 'prof_id' not in session: return jsonify(success=False, message="Not logged in"), 401
    d = request.get_json()
    student_id = d.get('student_id')
    status = d.get('status','present')
    if status not in ['present','absent']:
        return jsonify(success=False, message="Invalid status"), 400
    try:
        if status == 'present':
            mark_attendance(session_id, student_id, datetime.now().isoformat(), 1.0, 'present')
        else:
            update_attendance(session_id, student_id, 'absent')
        return jsonify(success=True, message=f"Student marked {status}")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/professor/session/<int:session_id>/manual-remove', methods=['POST'])
def professor_manual_remove(session_id):
    if 'prof_id' not in session: return jsonify(success=False, message="Not logged in"), 401
    d = request.get_json()
    try:
        delete_attendance(session_id, d.get('student_id'))
        return jsonify(success=True, message="Attendance record removed")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — HOD
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/hod/register_page')
def hod_register_page():
    return render_template_string(HOD_REG_PAGE, branches=BRANCHES)

@app.route('/hod/register', methods=['POST'])
def hod_register():
    d = request.get_json()
    for f in ['name','mobile','password','branch']:
        if not d.get(f): return jsonify(success=False, message=f"Missing: {f}")
    if len(d['password']) < 6: return jsonify(success=False, message="Password min 6 characters.")
    hid, err = register_hod(d['name'], d['mobile'], d['password'], d['branch'])
    if err: return jsonify(success=False, message=err)
    return jsonify(success=True, message=f"HOD '{d['name']}' registered for {d['branch']}!")

@app.route('/hod/login', methods=['GET', 'POST'])
def hod_login():
    if request.method == 'GET':
        return render_template_string(HOD_LOGIN_PAGE)
    d = request.get_json()
    row = get_hod_by_mobile(d.get('mobile',''), d.get('password',''))
    if not row: return jsonify(success=False, message="Invalid mobile or password.")
    session['hod_id']     = row[0]
    session['hod_name']   = row[1]
    session['hod_branch'] = row[2]
    return jsonify(success=True, message="Login successful!")

@app.route('/hod/logout')
def hod_logout():
    session.clear()
    return redirect(url_for('hod_login'))

@app.route('/hod/dashboard')
def hod_dashboard():
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    branch      = session['hod_branch']
    professors  = hod_get_professors(branch)
    sessions_db = hod_get_sessions(branch)
    student_sum = hod_student_attendance_summary(branch)
    students    = hod_get_students(branch)

    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name FROM students WHERE branch=?", (branch,))
    student_id_map = {row[1]: row[0] for row in c.fetchall()}
    conn.close()

    student_sum_with_ids = [row + (student_id_map.get(row[0], 0),) for row in student_sum]
    prof_session_counts  = {}
    for s in sessions_db:
        for pid, pname in professors:
            if pname == s[1]:
                prof_session_counts[pid] = prof_session_counts.get(pid, 0) + 1
    total_present = sum(r[4] for r in student_sum)
    total_records = sum(r[4]+r[5] for r in student_sum)
    avg_att = round(total_present/total_records*100, 1) if total_records > 0 else 0
    return render_template_string(HOD_DASH,
        branch=branch, hod_name=session['hod_name'],
        professors=professors, sessions=sessions_db,
        student_summary=student_sum_with_ids,
        total_students=len(students), total_professors=len(professors),
        total_sessions=len(sessions_db), avg_attendance=avg_att,
        prof_session_counts=prof_session_counts,
        years=YEARS)

@app.route('/hod/sessions')
def hod_sessions_api():
    if 'hod_branch' not in session: return jsonify(sessions=[])
    branch = session['hod_branch']
    pid    = request.args.get('professor_id')
    rows   = hod_get_sessions(branch, int(pid) if pid else None)
    data   = [{"id":r[0],"professor":r[1],"year":r[2],"branch":r[3],
               "subject":r[4],"start":r[5],"present":r[6],"absent":r[7]} for r in rows]
    return jsonify(sessions=data)

# ── NEW: student filter API ───────────────────────────────────────────────────
@app.route('/hod/students')
def hod_students_api():
    if 'hod_branch' not in session: return jsonify(students=[])
    branch = session['hod_branch']
    year = request.args.get('year', '')
    sort_by = request.args.get('sort_by', 'name')
    sort_order = request.args.get('sort_order', 'asc').lower()
    sort_order = 'DESC' if sort_order == 'desc' else 'ASC'
    sort_map = {
        'name': 'st.name',
        'year': 'st.year',
        'roll': 'CAST(st.roll_no AS INTEGER)',
        'present': 'present',
        'absent': 'absent',
    }
    sort_col = sort_map.get(sort_by, 'st.name')
    conn   = get_connection()
    c      = conn.cursor()
    query  = """
        SELECT st.id, st.name, st.year, st.div, st.roll_no, st.ien, st.mobile, st.email,
               COUNT(CASE WHEN a.status='present' THEN 1 END) as present,
               COUNT(CASE WHEN a.status='absent'  THEN 1 END) as absent
        FROM students st
        LEFT JOIN attendance a ON a.student_id = st.id
        WHERE st.branch=?
    """
    params = [branch]
    if year:
        query += " AND st.year=?"
        params.append(year)
    query += f" GROUP BY st.id ORDER BY {sort_col} {sort_order}, st.name ASC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    data = [{"id":r[0],"name":r[1],"year":r[2],"div":r[3],"roll":r[4],
             "ien":r[5],"mobile":r[6],"email":r[7],"present":r[8] or 0,"absent":r[9] or 0} for r in rows]
    return jsonify(students=data)


@app.route('/hod/professors')
def hod_professors_api():
    if 'hod_branch' not in session:
        return jsonify(professors=[])
    branch = session['hod_branch']
    sort_by = request.args.get('sort_by', 'name')
    sort_order = request.args.get('sort_order', 'asc').lower()
    sort_order = 'DESC' if sort_order == 'desc' else 'ASC'
    sort_map = {
        'name': 'p.name',
        'sessions': 'sessions_count',
        'mobile': 'p.mobile',
    }
    sort_col = sort_map.get(sort_by, 'p.name')

    conn = get_connection()
    c = conn.cursor()
    c.execute(f"""
        SELECT p.id, p.name, p.mobile, COUNT(DISTINCT s.id) AS sessions_count
        FROM professors p
        JOIN professor_branches pb ON pb.professor_id = p.id
        LEFT JOIN sessions s ON s.professor_id = p.id AND s.branch = ?
        WHERE pb.branch = ?
        GROUP BY p.id
        ORDER BY {sort_col} {sort_order}, p.name ASC
    """, (branch, branch))
    rows = c.fetchall()
    conn.close()

    data = [{"id": r[0], "name": r[1], "mobile": r[2], "sessions": r[3] or 0} for r in rows]
    return jsonify(professors=data)


@app.route('/hod/student/<int:student_id>/profile')
def hod_student_profile(student_id):
    if 'hod_branch' not in session:
        return jsonify(success=False, message="Not authenticated"), 401

    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT st.id, st.ien, st.name, st.email, st.mobile, st.branch, st.year, st.div, st.roll_no,
               COUNT(CASE WHEN a.status='present' THEN 1 END) AS present,
               COUNT(CASE WHEN a.status='absent' THEN 1 END) AS absent
        FROM students st
        LEFT JOIN attendance a ON a.student_id = st.id
        WHERE st.id = ?
        GROUP BY st.id
    """, (student_id,))
    row = c.fetchone()
    conn.close()

    if not row or row[5] != session['hod_branch']:
        return jsonify(success=False, message="Student not found for your branch"), 404

    profile = {
        "id": row[0],
        "ien": row[1],
        "name": row[2],
        "email": row[3],
        "mobile": row[4],
        "branch": row[5],
        "year": row[6],
        "div": row[7],
        "roll_no": row[8],
        "present": row[9] or 0,
        "absent": row[10] or 0,
    }
    return jsonify(success=True, profile=profile)


@app.route('/hod/professor/<int:pid>/profile')
def hod_professor_profile(pid):
    if 'hod_branch' not in session:
        return jsonify(success=False, message="Not authenticated"), 401

    branch = session['hod_branch']
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT p.id, p.name, p.mobile
        FROM professors p
        JOIN professor_branches pb ON pb.professor_id = p.id
        WHERE p.id = ? AND pb.branch = ?
    """, (pid, branch))
    prof = c.fetchone()

    if not prof:
        conn.close()
        return jsonify(success=False, message="Professor not found for your branch"), 404

    c.execute("""
        SELECT COUNT(DISTINCT s.id) AS sessions_count,
               COUNT(a.id) FILTER(WHERE a.status='present') AS present,
               COUNT(a.id) FILTER(WHERE a.status='absent')  AS absent
        FROM sessions s
        LEFT JOIN attendance a ON a.session_id = s.id
        WHERE s.professor_id = ? AND s.branch = ?
    """, (pid, branch))
    stats = c.fetchone()
    conn.close()

    profile = {
        "id": prof[0],
        "name": prof[1],
        "mobile": prof[2],
        "branches": get_professor_branches(pid),
        "sessions": stats[0] or 0,
        "present": stats[1] or 0,
        "absent": stats[2] or 0,
    }
    return jsonify(success=True, profile=profile)

# ── Export routes ─────────────────────────────────────────────────────────────

@app.route('/hod/export/sessions')
def hod_export_sessions():
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    branch = session['hod_branch']
    pid    = request.args.get('professor_id')
    rows_db = hod_get_sessions(branch, int(pid) if pid else None)
    rows = []
    for sid, pname, year, br, subject, start, present, absent in rows_db:
        total = present + absent
        pct   = round((present/total)*100, 1) if total > 0 else 0.0
        rows.append([sid, pname, year, br, subject, start.replace('T',' ')[:16], present, absent, pct])
    suffix = f"_prof_{pid}" if pid else "_all"
    return csv_response(
        f"hod_{branch}_sessions{suffix}.csv",
        ['Session ID','Professor','Year','Branch','Subject','Date & Time','Present','Absent','Attendance %'],
        rows)

@app.route('/hod/export/students')
def hod_export_students():
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    branch = session['hod_branch']
    year   = request.args.get('year','')
    conn   = get_connection()
    c      = conn.cursor()
    query  = """
        SELECT st.name, st.year, st.div, st.roll_no, st.ien,
               COUNT(CASE WHEN a.status='present' THEN 1 END) as present,
               COUNT(CASE WHEN a.status='absent'  THEN 1 END) as absent
        FROM students st
        LEFT JOIN attendance a ON a.student_id=st.id
        WHERE st.branch=?
    """
    params = [branch]
    if year:
        query += " AND st.year=?"
        params.append(year)
    query += " GROUP BY st.id ORDER BY st.year, st.name"
    c.execute(query, params)
    rows_db = c.fetchall()
    conn.close()
    rows = []
    for name, yr, div, roll, ien, present, absent in rows_db:
        total = (present or 0) + (absent or 0)
        pct   = round((present or 0)/total*100, 1) if total > 0 else 0.0
        rows.append([ien, name, yr, div or '—', roll, present or 0, absent or 0, pct])
    suffix = f"_{year}" if year else "_all"
    return csv_response(
        f"hod_{branch}_students{suffix}.csv",
        ['IEN','Name','Year','Division','Roll No','Present','Absent','Attendance %'],
        rows)

@app.route('/hod/export/professors')
def hod_export_professors():
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    branch = session['hod_branch']
    professors = hod_get_professors(branch)
    rows = []
    for pid, pname in professors:
        sessions_list = get_professor_sessions(pid, branch)
        total_present = sum(s['present'] for s in sessions_list)
        total_absent  = sum(s['absent']  for s in sessions_list)
        total         = total_present + total_absent
        pct           = round(total_present/total*100, 1) if total > 0 else 0.0
        rows.append([pid, pname, len(sessions_list), total_present, total_absent, pct])
    return csv_response(
        f"hod_{branch}_professors.csv",
        ['Professor ID','Name','Sessions Taken','Total Present','Total Absent','Overall Attendance %'],
        rows)

@app.route('/hod/professor/<int:pid>')
def hod_professor_detail(pid):
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    branch     = session['hod_branch']
    sessions_list = hod_get_sessions(branch, pid)
    professors = hod_get_professors(branch)
    pname      = next((p[1] for p in professors if p[0] == pid), "Unknown")
    rows_html  = ""
    for sid, pn, year, br, subject, start, present, absent in sessions_list:
        total = present + absent
        pct   = f"{(present/total*100):.1f}%" if total > 0 else "—"
        rows_html += f"<tr><td>{sid}</td><td>{year}</td><td>{subject}</td><td>{start[:16].replace('T',' ')}</td><td><span class='badge badge-present'>{present}</span></td><td><span class='badge badge-absent'>{absent}</span></td><td>{pct}</td></tr>"
    page = BASE_STYLE + NAV_HOD + f"""
    <div class="container">
      <div style="margin-bottom:16px"><a href="/hod/dashboard">← Back to Dashboard</a></div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
          <h2>Sessions by {pname} — {branch}</h2>
          <a href="/hod/export/sessions?professor_id={pid}" class="btn btn-success">⬇ Export CSV</a>
        </div>
        <table><thead><tr><th>#</th><th>Year</th><th>Subject</th><th>Date & Time</th><th>Present</th><th>Absent</th><th>%</th></tr></thead>
        <tbody>{rows_html or '<tr><td colspan="7" style="text-align:center;color:#666;padding:24px">No sessions.</td></tr>'}</tbody></table>
      </div>
    </div>"""
    return page

@app.route('/hod/student/<int:student_id>/delete', methods=['POST'])
def hod_delete_student(student_id):
    if 'hod_branch' not in session: return jsonify(success=False, message="Not authenticated"), 401
    try:
        conn = get_connection()
        c    = conn.cursor()
        c.execute("SELECT branch FROM students WHERE id=?", (student_id,))
        row = c.fetchone()
        if not row or row[0] != session['hod_branch']:
            conn.close()
            return jsonify(success=False, message="Cannot delete student from another branch"), 403
        c.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        c.execute("DELETE FROM students WHERE id=?", (student_id,))
        conn.commit()
        conn.close()
        return jsonify(success=True, message="Student deleted successfully")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/hod/student/<int:student_id>/attendance')
def hod_student_attendance(student_id):
    if 'hod_branch' not in session: return redirect(url_for('hod_login'))
    try:
        conn = get_connection()
        c    = conn.cursor()
        c.execute("SELECT name, roll_no, branch, year, div FROM students WHERE id=?", (student_id,))
        student = c.fetchone()
        if not student or student[2] != session['hod_branch']:
            conn.close()
            return redirect(url_for('hod_dashboard'))
        name, roll, branch, year, div = student
        c.execute("""
            SELECT s.subject,
                   COUNT(CASE WHEN a.status='present' THEN 1 END) as present,
                   COUNT(CASE WHEN a.status='absent'  THEN 1 END) as absent,
                   COUNT(a.id) as total
            FROM sessions s
            LEFT JOIN attendance a ON s.id=a.session_id AND a.student_id=?
            WHERE s.branch=? AND s.year=?
            GROUP BY s.subject ORDER BY s.subject
        """, (student_id, branch, year))
        subjects = c.fetchall()
        conn.close()

        subject_html   = ""
        total_present  = 0
        total_absent   = 0
        for subj, present, absent, total in subjects:
            if total == 0: continue
            total_present += present
            total_absent  += absent
            pct    = round((present/total)*100, 1) if total > 0 else 0
            color  = '#00c853' if pct >= 75 else '#ff9800' if pct >= 50 else '#ff5252'
            tcolor = '#00e676' if pct >= 75 else '#ffb74d' if pct >= 50 else '#ff5252'
            subject_html += f"""<tr>
                <td>{subj}</td>
                <td><span class="badge badge-present">{present}</span></td>
                <td><span class="badge badge-absent">{absent}</span></td>
                <td>{total}</td>
                <td><div style="display:flex;align-items:center;gap:8px">
                  <div class="progress-bar" style="width:100px;height:8px">
                    <div class="progress-fill" style="width:{pct}%;background:{color}"></div>
                  </div>
                  <span style="font-size:.82rem;color:{tcolor}">{pct}%</span>
                </div></td>
            </tr>"""
        overall_total = total_present + total_absent
        overall_pct   = round((total_present/overall_total)*100, 1) if overall_total > 0 else 0
        oc = '#00c853' if overall_pct >= 75 else '#ff9800' if overall_pct >= 50 else '#ff5252'

        page = BASE_STYLE + NAV_HOD + f"""
        <div class="container">
          <div style="margin-bottom:16px"><a href="/hod/dashboard">← Back to Dashboard</a></div>
          <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
              <div>
                <h2 style="margin-bottom:8px">{name}</h2>
                <span style="color:#aaa;font-size:.9rem">Roll: {roll} | Year: {year} | Division: {div or '—'}</span>
              </div>
            </div>
            <div class="stat-grid">
              <div class="stat-card"><div class="num" style="color:#00e676">{total_present}</div><div class="lbl">Total Present</div></div>
              <div class="stat-card"><div class="num" style="color:#ff5252">{total_absent}</div><div class="lbl">Total Absent</div></div>
              <div class="stat-card"><div class="num">{overall_total}</div><div class="lbl">Total Sessions</div></div>
              <div class="stat-card"><div class="num" style="color:{oc}">{overall_pct}%</div><div class="lbl">Overall</div></div>
            </div>
            <h3 style="margin-bottom:16px">Subject-wise Attendance</h3>
            <table>
              <thead><tr><th>Subject</th><th>Present</th><th>Absent</th><th>Total</th><th>Attendance %</th></tr></thead>
              <tbody>{subject_html or '<tr><td colspan="5" style="text-align:center;color:#666;padding:24px">No records found.</td></tr>'}</tbody>
            </table>
          </div>
        </div>"""
        return page
    except Exception as e:
        return f"Error: {str(e)}", 500


# ─────────────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/check_face", methods=["POST"])
def check_face():
    d = request.get_json()
    b64 = d.get("image","")
    if not b64: return jsonify(found=False)
    try:
        _, data = b64.split(",", 1)
        arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: return jsonify(found=False)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        return jsonify(found=len(locs) == 1)
    except Exception:
        return jsonify(found=False)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nWeb App running:")
    print("  Student Registration  : http://localhost:5000/student")
    print("  Professor Registration: http://localhost:5000/professor/register")
    print("  Professor Login       : http://localhost:5000/professor/login")
    print("  HOD Registration      : http://localhost:5000/hod/register_page")
    print("  HOD Login             : http://localhost:5000/hod/login")
    print("  Student Login         : http://localhost:5000/student/login")
    print("\nFor mobile access: run 'ipconfig' and use http://YOUR_IP:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host='0.0.0.0', port=5000)