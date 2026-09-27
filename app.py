"""
SetuNER — SIH 26002: AI-Based Smart Logistics & Accessibility Intelligence
Platform for the North Eastern Region (NER)
=============================================================================
Single-file prototype implementing the full 36-hour MVP plan with local,
self-contained stand-ins wherever the plan calls for a live external
government API (IMD, MOSDAC, Bhuvan, OSRM) — per the plan's own guardrail:
"use sample/static data when access is unavailable." Nothing here silently
pretends to be a live feed; every stand-in is labeled as such in the UI.

WHAT'S IN HERE
  - Auth / role-select portal, with a 3-language toggle (EN/HI/AS) as a
    concrete first step on the "multilingual coverage" gap, not just a claim
  - SetuNER GIS Command Centre: pure-SVG map, no Leaflet/CDN/tiles, so the
    whole dashboard works with zero internet access
  - A real ML risk model (scikit-learn RandomForest, trained on a synthetic
    but internally-consistent dataset, evaluated with stratified k-fold CV)
    running ALONGSIDE the rule-based engine as a cross-check — if scikit-
    learn isn't installed, or the model's confidence is low, the system
    automatically falls back to the hard rule-based score. This mirrors the
    plan's own guardrail: "small ML model plus hard safety rules."
  - A denser, fully-connected corridor network spanning the NER, so routing
    is genuinely multi-hop (a stand-in for OSRM, since we can't reach the
    live OSRM service from here)
  - Offline-first field reporting: reports queue in the browser and sync
    once connectivity returns. This is accurately store-and-forward to a
    central server, NOT device-to-device mesh networking — renamed here to
    stop overclaiming a capability (true peer relay would need Bluetooth/
    LoRa hardware this prototype doesn't have)
  - A native (no CDN) SVG analytics panel — a deliberate deviation from the
    plan's suggested Chart.js, so analytics keep working fully offline too
  - Vehicle simulation with a live ETA, and a one-click scripted
    "medicine truck" demo walking the plan's exact 7-step story end-to-end
  - "Simulate Weather" is explicitly labeled DEV/DEMO ONLY in the UI — this
    is the prototype's "God Mode" equivalent, scoped per the review's
    security note (never ship this panel in a production build)
  - A read-only interoperability export endpoint, as a first sketch of the
    NDMA/SDMA integration story

RUN:
    1. pip install flask
       (optional, for the ML layer: pip install scikit-learn)
       (some Linux setups: add --break-system-packages to either)
    2. python app.py
    3. Open http://localhost:5000

setuner.db and an uploads/ folder are created automatically on first run.
Delete setuner.db any time to reset demo data.
"""
import os
import sys
import math
import random
import sqlite3
import subprocess
import time
from flask import Flask, g, jsonify, request, Response

DB_PATH = "setuner.db"
UPLOAD_DIR = "uploads"
ROUTING_SCRIPT = "setu3.py"

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# ML risk model — scikit-learn if available, hard rule-based fallback if not
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

ML_MODEL = None
MODEL_METRICS = {"available": False}


def synthetic_training_data(n=400, seed=42):
    """Generates a synthetic but internally-consistent dataset for the demo:
    features are terrain risk, historical disruption count, and rainfall;
    the label follows the same underlying signal as the rule engine plus
    noise, so the ML model has something real to learn without needing a
    live IMD/GSI feed. Class weights are skewed toward 'low' on purpose —
    severe landslide events are rare in real data too (class imbalance is
    part of the point, not an oversight)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        terrain = rng.randint(5, 55)
        hist = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[28, 24, 18, 12, 9, 5, 4])[0]
        rainfall = rng.choices(range(0, 200), k=1)[0]
        noise = rng.gauss(0, 8)
        raw = terrain + hist * 3 + rainfall * 0.4 + noise
        if raw >= 70:
            label = "high"
        elif raw >= 40:
            label = "medium"
        else:
            label = "low"
        X.append([terrain, hist, rainfall])
        y.append(label)
    return X, y


def train_ml_model():
    """Trains the RandomForest and evaluates it with stratified 5-fold CV —
    this is the model-evaluation methodology the review flagged as missing.
    Metrics are macro-averaged because of the class imbalance above."""
    global ML_MODEL, MODEL_METRICS
    if not ML_AVAILABLE:
        MODEL_METRICS = {"available": False, "reason": "scikit-learn not installed — running on rule-based fallback only."}
        return

    X, y = synthetic_training_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    try:
        y_pred = cross_val_predict(model, X, y, cv=skf)
        MODEL_METRICS = {
            "available": True,
            "cv_folds": 5,
            "accuracy": round(accuracy_score(y, y_pred), 3),
            "precision_macro": round(precision_score(y, y_pred, average="macro", zero_division=0), 3),
            "recall_macro": round(recall_score(y, y_pred, average="macro", zero_division=0), 3),
            "training_samples": len(X),
            "note": "Trained on a synthetic dataset (no live IMD/GSI feed in this environment) — same feature schema as production would use, so the pipeline is real even though this training data isn't.",
        }
    except Exception as e:
        MODEL_METRICS = {"available": False, "reason": f"Training failed: {e}"}
        return

    model.fit(X, y)
    ML_MODEL = model


def ml_predict_level(terrain_risk, historical_disruptions, rainfall_mm, confidence_floor=0.45):
    """Returns (level, confidence, source). Falls back to the rule-based
    level whenever the model is unavailable or unsure — this is the 'hard
    safety rule' backstop the plan calls for, so a shaky ML call never
    silently overrides a known-risk corridor."""
    if ML_MODEL is not None:
        try:
            proba = ML_MODEL.predict_proba([[terrain_risk, historical_disruptions, rainfall_mm]])[0]
            classes = ML_MODEL.classes_
            best_idx = max(range(len(proba)), key=lambda i: proba[i])
            label, confidence = classes[best_idx], round(float(proba[best_idx]), 2)
            if confidence >= confidence_floor:
                return label, confidence, "model"
        except Exception:
            pass
    # fallback: same thresholds as the rule engine
    raw = terrain_risk + historical_disruptions * 3 + rainfall_mm * 0.4
    if raw >= 70:
        level = "high"
    elif raw >= 40:
        level = "medium"
    else:
        level = "low"
    return level, None, "fallback_rules"


# ---------------------------------------------------------------------------
# PORTAL — auth / role-select shell + GIS dashboard + incident reporter
# ---------------------------------------------------------------------------
PORTAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Logistics &amp; Emergency Response Portal</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
  <style>
:root {
  --primary-color: #2563eb;
  --primary-hover: #1d4ed8;
  --bg-color: #f8fafc;
  --card-bg: #ffffff;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --border-color: #e2e8f0;
  --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
}

* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background-color: var(--bg-color); color: var(--text-main); min-height: 100vh; }
.hidden { display: none !important; }

.auth-wrapper { display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
.auth-card { background: var(--card-bg); width: 100%; max-width: 440px; padding: 32px; border-radius: 16px; box-shadow: var(--shadow); border: 1px solid var(--border-color); }
.auth-card.wide { max-width: 640px; }
.auth-header { text-align: center; margin-bottom: 16px; }
.auth-header-top { display: flex; justify-content: flex-end; margin-bottom: 10px; }
.logo-icon { font-size: 2.5rem; color: var(--primary-color); margin-bottom: 12px; }
.lang-select { font-size: 0.75rem; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--border-color); background: white; cursor: pointer; }

.tabs { display: flex; background: #f1f5f9; padding: 4px; border-radius: 8px; margin-bottom: 24px; }
.tab-btn { flex: 1; padding: 10px; border: none; background: transparent; font-weight: 500; color: var(--text-muted); cursor: pointer; border-radius: 6px; transition: all 0.2s ease; }
.tab-btn.active { background: white; color: var(--primary-color); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

.auth-form { display: none; flex-direction: column; gap: 16px; }
.auth-form.active-form { display: flex; }

.info-box {
  display: flex; align-items: center; gap: 12px; background: #eff6ff; border: 1px solid #bfdbfe;
  padding: 12px 16px; border-radius: 8px; color: #1e40af; font-size: 0.875rem;
}

.input-group { display: flex; flex-direction: column; gap: 6px; }
.input-field { display: flex; align-items: center; border: 1px solid var(--border-color); border-radius: 8px; padding: 0 12px; background: #f8fafc; }
.input-field i { color: var(--text-muted); margin-right: 10px; }
.input-field input, select { width: 100%; padding: 12px 0; border: none; background: transparent; outline: none; }
select { border: 1px solid var(--border-color); padding: 12px; border-radius: 8px; }

.btn-primary { background: var(--primary-color); color: white; padding: 12px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s ease; margin-top: 8px; }
.btn-primary:hover { background: var(--primary-hover); }
.btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }

.app-viewport { display: flex; flex-direction: column; height: 100vh; }
.top-nav { height: 60px; background: white; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; padding: 0 24px; }
.nav-brand { font-weight: 700; font-size: 1.1rem; display: flex; align-items: center; gap: 10px; color: var(--primary-color); }
.nav-tabs { display: flex; gap: 8px; }
.nav-tabs button { border: 1px solid var(--border-color); background: white; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 500; color: var(--text-muted); }
.nav-tabs button.active { background: var(--primary-color); color: white; border-color: var(--primary-color); }
.user-info { display: flex; align-items: center; gap: 12px; }
.badge { background: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.btn-logout { background: transparent; border: 1px solid var(--border-color); padding: 6px 12px; border-radius: 6px; cursor: pointer; }
.view-panel { flex: 1; padding: 24px; overflow-y: auto; }
.map-placeholder { height: 80%; background: #e2e8f0; border-radius: 12px; overflow: hidden; border: 1px solid #cbd5e1; margin-top: 20px; }
.map-placeholder iframe { width: 100%; height: 100%; border: 0; }
.form-card { background: white; padding: 24px; border-radius: 12px; border: 1px solid var(--border-color); max-width: 600px; margin-top: 20px; }
.gps-field { display: flex; gap: 10px; }
.gps-field input { border: 1px solid var(--border-color); padding: 10px; border-radius: 8px; flex: 1; }
.btn-secondary { background: #f1f5f9; border: 1px solid var(--border-color); padding: 10px 14px; border-radius: 8px; cursor: pointer; }
.status-msg { margin-top: 12px; font-size: 0.875rem; }
.status-msg.ok { color: #16a34a; }
.status-msg.err { color: #dc2626; }

.route-map { border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden; margin: 14px 0 10px; background: #eaf1f8; }
.route-map svg { width: 100%; height: auto; display: block; }
.route-legend { display: flex; gap: 16px; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 10px; }
.legend-dot { display: inline-block; width: 14px; height: 4px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.legend-dot.primary { background: #2563eb; }
.legend-dot.alt { background: #f59e0b; }
.route-block { margin-bottom: 16px; }
.route-block h4 { font-size: 0.9rem; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.route-tag { font-size: 0.65rem; text-transform: uppercase; font-weight: 700; background: #dbeafe; color: #1d4ed8; padding: 2px 8px; border-radius: 10px; }
.route-tag.alt-tag { background: #fef3c7; color: #b45309; }
.muted-note { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 10px; }
.spof-box { border: 1px solid #fecaca; background: #fef2f2; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
.spof-box .spof-title { font-weight: 700; color: #b91c1c; font-size: 0.9rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
.spof-box p { font-size: 0.8rem; color: #7f1d1d; margin-bottom: 8px; }
.spof-box ul { padding-left: 18px; font-size: 0.8rem; color: #7f1d1d; }
.spof-box li { margin-bottom: 4px; }
.route-summary { display: flex; gap: 18px; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 10px; }
.route-summary i { margin-right: 4px; color: var(--primary-color); }
.route-steps { display: flex; flex-direction: column; gap: 8px; }
.route-step { display: flex; align-items: flex-start; gap: 10px; background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 10px; text-align: left; }
.step-num { flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%; background: var(--primary-color); color: #fff; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.step-name { font-weight: 600; font-size: 0.875rem; }
.step-meta { font-size: 0.75rem; color: var(--text-muted); }
.step-meta .level-high { color: #c62828; font-weight: 700; }
.step-meta .level-medium { color: #ef6c00; font-weight: 700; }
.step-meta .level-low { color: #2e7d32; font-weight: 700; }
.eta-box { display: flex; gap: 18px; align-items: center; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 10px 14px; margin-top: 12px; font-size: 0.85rem; color: #1e40af; }
.eta-box b { font-size: 1rem; }
  </style>
</head>
<body>

  <!-- AUTH CONTAINER -->
  <div class="auth-wrapper" id="authWrapper">
    <div class="auth-card">
      <div class="auth-header">
        <div class="auth-header-top">
          <select class="lang-select" id="langSelect">
            <option value="en">EN</option>
            <option value="hi">HI</option>
            <option value="as">AS</option>
          </select>
        </div>
        <i class="fa-solid fa-truck-fast logo-icon"></i>
        <h2 data-i18n="title">Fleet &amp; Field Portal</h2>
        <p data-i18n="tagline">Select your option to proceed</p>
      </div>

      <div class="tabs">
        <button class="tab-btn active" id="tabAdminBtn" onclick="switchTab('admin')" data-i18n="tabAdmin">Admin / Staff</button>
        <button class="tab-btn" id="tabTruckBtn" onclick="switchTab('truck')" data-i18n="tabTruck">Truck Navigation</button>
      </div>

      <!-- Form 1: Admin / Staff Login -->
      <form id="adminLoginForm" class="auth-form active-form" onsubmit="handleAdminLogin(event)">
        <div class="input-group">
          <label for="adminEmail" data-i18n="emailLabel">Email Address</label>
          <div class="input-field">
            <i class="fa-regular fa-envelope"></i>
            <input type="email" id="adminEmail" placeholder="admin@organization.com" required />
          </div>
        </div>

        <div class="input-group">
          <label for="adminPassword" data-i18n="passwordLabel">Password</label>
          <div class="input-field">
            <i class="fa-solid fa-lock"></i>
            <input type="password" id="adminPassword" placeholder="••••••••" required />
          </div>
        </div>

        <button type="submit" class="btn-primary" data-i18n="signIn">Sign In as Admin / Staff</button>
        <p class="status-msg" id="adminLoginStatus"></p>
      </form>

      <!-- Form 2: Route Planner (No ID/PIN required) -->
      <div id="truckLoginForm" class="auth-form">
        <div class="info-box">
          <i class="fa-solid fa-circle-info"></i>
          <p data-i18n="truckInfo">Pick an origin and destination — the app calculates the lowest-risk route across the current corridor network before you head out.</p>
        </div>

        <div class="input-group">
          <label for="routeOrigin" data-i18n="originLabel">Origin</label>
          <select id="routeOrigin"></select>
        </div>
        <div class="input-group">
          <label for="routeDestination" data-i18n="destLabel">Destination</label>
          <select id="routeDestination"></select>
        </div>

        <button type="button" class="btn-primary" id="findRouteBtn" onclick="findOptimalRoute()">
          <i class="fa-solid fa-route"></i> <span data-i18n="findRoute">Find Optimized Route</span>
        </button>
        <p class="status-msg" id="routeFindStatus"></p>

        <div id="routeResult" class="hidden"></div>
      </div>
    </div>
  </div>

  <!-- MAIN APP CONTAINER -->
  <div class="app-viewport hidden" id="appViewport">
    <header class="top-nav">
      <div class="nav-brand">
        <i class="fa-solid fa-shield-halved"></i>
        <span id="portalTitle">Portal View</span>
      </div>
      <div class="nav-tabs">
        <button id="navGisBtn" class="active" onclick="switchView('gis')">GIS Dashboard</button>
        <button id="navReportBtn" onclick="switchView('report')">Report Incident</button>
      </div>
      <div class="user-info">
        <span id="userBadge" class="badge">Role</span>
        <button class="btn-logout" onclick="logout()"><i class="fa-solid fa-right-from-bracket"></i> Logout</button>
      </div>
    </header>

    <!-- GIS Dashboard -->
    <main id="gisDashboardView" class="view-panel">
      <div class="panel-header">
        <h2>GIS Logistics Dashboard</h2>
        <p>Real-time vehicle tracking, fleet monitoring, and spatial analysis</p>
      </div>
      <div class="map-placeholder">
        <iframe src="/setuner" title="SetuNER Command Centre"></iframe>
      </div>
    </main>

    <!-- Incident Reporter -->
    <main id="reporterView" class="view-panel hidden">
      <div class="panel-header">
        <h2 data-i18n="reportHeading">Report Natural Hazard Incident</h2>
        <p data-i18n="reportSub">Submit GPS location and evidence for landslides or floods</p>
      </div>
      <div class="form-card">
        <form onsubmit="handleReportSubmit(event)">
          <div class="input-group">
            <label>Incident Type</label>
            <select id="incidentType" required>
              <option value="">Select Incident Type</option>
              <option value="landslide">Landslide</option>
              <option value="flood">Flood / Inundation</option>
              <option value="roadblock">Road Damage / Blockage</option>
            </select>
          </div>
          <div class="input-group">
            <label>GPS Coordinates</label>
            <div class="gps-field">
              <input type="text" id="gpsCoords" placeholder="Latitude, Longitude" readonly required />
              <button type="button" class="btn-secondary" onclick="getGPSLocation()">
                <i class="fa-solid fa-location-crosshairs"></i> Get Current Location
              </button>
            </div>
          </div>
          <div class="input-group">
            <label>Upload Incident Photo</label>
            <input type="file" id="disasterImage" accept="image/*" capture="environment" required />
          </div>
          <button type="submit" class="btn-primary">Submit Report</button>
          <p class="status-msg" id="reportStatus"></p>
        </form>
      </div>
    </main>
  </div>

<script>
// ---- i18n (concrete step on the multilingual gap: EN/HI/AS now, full NER
// language coverage is the phased rollout the review recommended) ----
const I18N = {
  en: { title: "Fleet & Field Portal", tagline: "Select your option to proceed", tabAdmin: "Admin / Staff", tabTruck: "Truck Navigation", emailLabel: "Email Address", passwordLabel: "Password", signIn: "Sign In as Admin / Staff", truckInfo: "Pick an origin and destination — the app calculates the lowest-risk route across the current corridor network before you head out.", originLabel: "Origin", destLabel: "Destination", findRoute: "Find Optimized Route", reportHeading: "Report Natural Hazard Incident", reportSub: "Submit GPS location and evidence for landslides or floods" },
  hi: { title: "फ्लीट और फील्ड पोर्टल", tagline: "आगे बढ़ने के लिए अपना विकल्प चुनें", tabAdmin: "एडमिन / स्टाफ", tabTruck: "ट्रक नेविगेशन", emailLabel: "ईमेल पता", passwordLabel: "पासवर्ड", signIn: "एडमिन/स्टाफ के रूप में साइन इन करें", truckInfo: "मूल और गंतव्य चुनें — ऐप रवाना होने से पहले सबसे कम जोखिम वाला मार्ग निकालेगा।", originLabel: "मूल स्थान", destLabel: "गंतव्य", findRoute: "अनुकूलित मार्ग खोजें", reportHeading: "प्राकृतिक आपदा घटना दर्ज करें", reportSub: "भूस्खलन या बाढ़ के लिए जीपीएस स्थान और प्रमाण जमा करें" },
  as: { title: "ফ্লীট আৰু ফিল্ড পৰ্টেল", tagline: "আগবাঢ়িবলৈ আপোনাৰ বিকল্প বাছক", tabAdmin: "প্ৰশাসক / কৰ্মী", tabTruck: "ট্ৰাক নেভিগেশ্যন", emailLabel: "ইমেইল ঠিকনা", passwordLabel: "পাছৱৰ্ড", signIn: "প্ৰশাসক/কৰ্মী হিচাপে ছাইন ইন কৰক", truckInfo: "উৎপত্তিস্থান আৰু গন্তব্য বাছক — এপটোৱে যাত্ৰাৰ আগতে আটাইতকৈ কম বিপদাশংকাযুক্ত পথ গণনা কৰিব।", originLabel: "উৎপত্তিস্থান", destLabel: "গন্তব্য", findRoute: "অনুকূলিত পথ বিচাৰক", reportHeading: "প্ৰাকৃতিক দুৰ্যোগ ঘটনা প্ৰতিবেদন কৰক", reportSub: "গৃহস্খলন বা বানপানীৰ বাবে GPS অৱস্থান আৰু প্ৰমাণ দাখিল কৰক" }
};
function applyLang(lang) {
  const t = I18N[lang] || I18N.en;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (t[key]) el.textContent = t[key];
  });
}
document.getElementById('langSelect').addEventListener('change', e => applyLang(e.target.value));

// ---- tab / view switching ----
function switchTab(type) {
  document.getElementById('tabAdminBtn').classList.toggle('active', type === 'admin');
  document.getElementById('tabTruckBtn').classList.toggle('active', type === 'truck');
  document.getElementById('adminLoginForm').classList.toggle('active-form', type === 'admin');
  document.getElementById('truckLoginForm').classList.toggle('active-form', type === 'truck');
  document.querySelector('.auth-card').classList.toggle('wide', type === 'truck');
}

function switchView(view) {
  document.getElementById('gisDashboardView').classList.toggle('hidden', view !== 'gis');
  document.getElementById('reporterView').classList.toggle('hidden', view !== 'report');
  document.getElementById('navGisBtn').classList.toggle('active', view === 'gis');
  document.getElementById('navReportBtn').classList.toggle('active', view === 'report');
}

// ---- admin / staff login ----
// NOTE: no real backend auth endpoint exists in this prototype — this only
// gates the UI locally so the dashboard is reachable for the demo. Wire a
// real /api/login check before this goes anywhere near production.
function handleAdminLogin(event) {
  event.preventDefault();
  const email = document.getElementById('adminEmail').value;
  document.getElementById('authWrapper').classList.add('hidden');
  document.getElementById('appViewport').classList.remove('hidden');
  document.getElementById('portalTitle').textContent = 'Admin / Staff Portal';
  document.getElementById('userBadge').textContent = email.split('@')[0] || 'Staff';
  switchView('gis');
}

function logout() {
  document.getElementById('appViewport').classList.add('hidden');
  document.getElementById('authWrapper').classList.remove('hidden');
  document.getElementById('adminLoginForm').reset();
  document.getElementById('adminLoginStatus').textContent = '';
}

// ---- route planner (truck navigation) ----
const ROUTE_BOUNDS = { latMin: 22.8, latMax: 28.0, lngMin: 87.7, lngMax: 95.5, W: 640, H: 560, M: 40 };

function routeProject(lat, lng) {
  const { latMin, latMax, lngMin, lngMax, W, H, M } = ROUTE_BOUNDS;
  const x = M + ((lng - lngMin) / (lngMax - lngMin)) * (W - 2 * M);
  const y = M + ((latMax - lat) / (latMax - latMin)) * (H - 2 * M);
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

// Alternates share the exact endpoints of their primary corridor, so a
// straight line would sit directly on top of it — curve each one away from
// the straight line so it reads as its own distinct road.
function routeCurvedPathD(x1, y1, x2, y2, offset) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const nx = -dy / len, ny = dx / len;
  const cx = mx + nx * offset, cy = my + ny * offset;
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
}

async function populateRouteSelectors() {
  try {
    const res = await fetch('/api/places');
    const data = await res.json();
    const originSel = document.getElementById('routeOrigin');
    const destSel = document.getElementById('routeDestination');
    const opts = data.places.map(p => `<option value="${p}">${p}</option>`).join('');
    originSel.innerHTML = opts;
    destSel.innerHTML = opts;
    if (data.places.length > 1) destSel.selectedIndex = 1;
  } catch (e) { /* selects stay empty; user can retry after reload */ }
}

function buildRouteMapSVG(allCorridors, hospitals, primaryIds, altIds) {
  const { W, H } = ROUTE_BOUNDS;
  const parts = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">`);
  parts.push(`<rect x="1" y="1" width="${W - 2}" height="${H - 2}" fill="#eaf1f8" stroke="#c9d8e6" rx="12"/>`);

  const placeSeen = new Set();
  const others = allCorridors.filter(c => !primaryIds.has(c.id) && !altIds.has(c.id));
  const alt = allCorridors.filter(c => altIds.has(c.id) && !primaryIds.has(c.id));
  const primary = allCorridors.filter(c => primaryIds.has(c.id));

  [...others, ...alt, ...primary].forEach(c => {
    const isPrimary = primaryIds.has(c.id);
    const isAlt = altIds.has(c.id) && !isPrimary;
    const [x1, y1] = routeProject(c.from[0], c.from[1]);
    const [x2, y2] = routeProject(c.to[0], c.to[1]);
    let color = '#c7d2e0', width = 3, dash = '';
    // curve every road that has a same-endpoint counterpart (i.e. is itself an
    // alternate, or its is_alternate_for id is highlighted) away from the
    // straight line, so overlapping road pairs stay visually distinct
    const offset = c.is_alternate_for ? -22 : (8 + (c.id % 3) * 3);
    const d = routeCurvedPathD(x1, y1, x2, y2, offset);
    if (isPrimary) { color = '#2563eb'; width = 6; }
    else if (isAlt) { color = '#f59e0b'; width = 5; dash = 'stroke-dasharray="2,7"'; }
    parts.push(`<path id="rline-${c.id}" d="${d}" fill="none" stroke="#ffffff" stroke-width="${width + 3}" stroke-linecap="round" opacity="0.85"/>`);
    parts.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" ${dash}/>`);

    [[c.from, c.from_place], [c.to, c.to_place]].forEach(([coord, label]) => {
      const cleanLabel = label.replace(' (Alt)', '');
      const key = coord.join(',');
      if (placeSeen.has(key)) return;
      placeSeen.add(key);
      const [px, py] = routeProject(coord[0], coord[1]);
      const textW = cleanLabel.length * 6.2 + 6;
      parts.push(`<rect x="${px + 6}" y="${py - 19}" width="${textW}" height="14" rx="3" fill="#ffffff" opacity="0.82"/>`);
      parts.push(`<circle cx="${px}" cy="${py}" r="5" fill="#ffffff" stroke="#1c1c1c" stroke-width="2"/>`);
      parts.push(`<circle cx="${px}" cy="${py}" r="2" fill="#1c1c1c"/>`);
      parts.push(`<text x="${px + 9}" y="${py - 8}" font-size="12" font-weight="700" fill="#0f172a">${cleanLabel}</text>`);
    });
  });

  hospitals.forEach(h => {
    const [x, y] = routeProject(h.lat, h.lng);
    parts.push(`<circle cx="${x}" cy="${y}" r="7" fill="#fff" stroke="#c62828" stroke-width="2"/>`);
    parts.push(`<line x1="${x - 3.5}" y1="${y}" x2="${x + 3.5}" y2="${y}" stroke="#c62828" stroke-width="2"/>`);
    parts.push(`<line x1="${x}" y1="${y - 3.5}" x2="${x}" y2="${y + 3.5}" stroke="#c62828" stroke-width="2"/>`);
  });

  parts.push(`<circle id="vehicleMarker" cx="-100" cy="-100" r="7" fill="#16a34a" stroke="#fff" stroke-width="2" class="hidden"/>`);
  parts.push('</svg>');
  return parts.join('');
}

async function findOptimalRoute() {
  const origin = document.getElementById('routeOrigin').value;
  const destination = document.getElementById('routeDestination').value;
  const status = document.getElementById('routeFindStatus');
  const resultBox = document.getElementById('routeResult');

  if (!origin || !destination) {
    status.textContent = 'Choose both an origin and a destination.';
    status.className = 'status-msg err';
    return;
  }
  if (origin === destination) {
    status.textContent = 'Origin and destination must be different.';
    status.className = 'status-msg err';
    return;
  }

  status.textContent = 'Calculating optimized route...';
  status.className = 'status-msg';
  resultBox.classList.add('hidden');

  try {
    const res = await fetch('/api/optimize-route', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, destination })
    });
    const data = await res.json();
    if (data.status !== 'success') {
      status.textContent = data.message || 'No route found.';
      status.className = 'status-msg err';
      return;
    }
    status.textContent = '';
    await renderRouteResult(data.route);
    resultBox.classList.remove('hidden');
  } catch (e) {
    status.textContent = 'Could not reach the server.';
    status.className = 'status-msg err';
  }
}

let currentRouteForSim = null;

async function renderRouteResult(routeData) {
  const cRes = await fetch('/api/corridors');
  const cdata = await cRes.json();
  const primary = routeData.primary;
  const alternate = routeData.alternate;
  const primaryIds = new Set(primary.segments.map(s => s.corridor.id));
  const altIds = new Set(alternate ? alternate.segments.map(s => s.corridor.id) : []);
  const mapHtml = buildRouteMapSVG(cdata.corridors, cdata.hospitals, primaryIds, altIds);
  currentRouteForSim = primary;

  const stepsHtml = (segments) => segments.map((s, i) => `
    <div class="route-step">
      <span class="step-num">${i + 1}</span>
      <div>
        <div class="step-name">${s.corridor.name}</div>
        <div class="step-meta">${s.place_from} &rarr; ${s.place_to} &middot; ${s.distance_km} km &middot; risk <span class="level-${s.corridor.level}">${s.corridor.score} (${s.corridor.level})</span></div>
      </div>
    </div>
  `).join('');

  let html = `
    <div class="route-map">${mapHtml}</div>
    <div class="route-legend">
      <span><i class="legend-dot primary"></i>Recommended route</span>
      ${alternate ? '<span><i class="legend-dot alt"></i>Alternate route</span>' : ''}
    </div>
    <div class="route-block">
      <h4>Recommended Route <span class="route-tag">Lowest risk</span></h4>
      <div class="route-summary">
        <span><i class="fa-solid fa-road"></i>${primary.total_distance_km} km</span>
        <span><i class="fa-solid fa-triangle-exclamation"></i>Route risk score: ${primary.total_risk_score}</span>
      </div>
      <div class="route-steps">${stepsHtml(primary.segments)}</div>
      <div class="eta-box">
        <span><i class="fa-solid fa-truck"></i> Est. travel time at 40 km/h: <b id="etaStatic">${(primary.total_distance_km / 40).toFixed(1)} hrs</b></span>
        <button type="button" class="btn-secondary" onclick="startVehicleSimulation()"><i class="fa-solid fa-play"></i> Simulate Dispatch</button>
      </div>
      <div id="simLog" class="muted-note"></div>
    </div>
  `;

  if (alternate) {
    html += `
      <div class="route-block">
        <h4>Alternate Route <span class="route-tag alt-tag">Backup option</span></h4>
        <div class="route-summary">
          <span><i class="fa-solid fa-road"></i>${alternate.total_distance_km} km</span>
          <span><i class="fa-solid fa-triangle-exclamation"></i>Route risk score: ${alternate.total_risk_score}</span>
        </div>
        <div class="route-steps">${stepsHtml(alternate.segments)}</div>
      </div>
    `;
  } else if (routeData.advisory) {
    const adv = routeData.advisory;
    const rally = adv.nearest_rally_point
      ? `${adv.nearest_rally_point.name} (${adv.nearest_rally_point.distance_km} km from destination)`
      : 'No hospital data available for this area';
    html += `
      <div class="spof-box">
        <div class="spof-title"><i class="fa-solid fa-triangle-exclamation"></i> Critical Corridor — No Alternate</div>
        <p>${adv.message}</p>
        <ul>
          <li><b>Nearest emergency rally point:</b> ${rally}</li>
          <li><b>Estimated clearance time (from history):</b> ~${adv.estimated_clearance_hours} hours</li>
        </ul>
      </div>
    `;
  }

  html += `
    <button type="button" class="btn-secondary full" id="launchRouteBtn" style="margin-top:14px;" onclick="launchPythonRoutingApp()">
      <i class="fa-solid fa-diamond-turn-right"></i> Launch Turn-by-Turn Navigation App
    </button>
    <p class="status-msg" id="routeLaunchStatus"></p>
  `;

  document.getElementById('routeResult').innerHTML = html;
}

// ---- vehicle dispatch simulation with live ETA (client-side; time is
// compressed for the demo, not real elapsed time) ----
function startVehicleSimulation() {
  if (!currentRouteForSim) return;
  const marker = document.getElementById('vehicleMarker');
  const log = document.getElementById('simLog');
  if (!marker) return;
  marker.classList.remove('hidden');
  const segments = currentRouteForSim.segments;
  const totalKm = currentRouteForSim.total_distance_km;
  const speedKmh = 40;
  const totalHours = totalKm / speedKmh;
  log.innerHTML = '';

  let segIdx = 0;
  function animateSegment() {
    if (segIdx >= segments.length) {
      log.innerHTML += `<div>&#9989; Arrived at destination. Total time: ${totalHours.toFixed(1)} hrs at ${speedKmh} km/h.</div>`;
      return;
    }
    const seg = segments[segIdx];
    const corridor = seg.corridor;
    const fromCoord = corridor.from_place === seg.place_from ? corridor.from : corridor.to;
    const toCoord = corridor.from_place === seg.place_from ? corridor.to : corridor.from;
    const [x1, y1] = routeProject(fromCoord[0], fromCoord[1]);
    const [x2, y2] = routeProject(toCoord[0], toCoord[1]);
    // follow the same curve the road is drawn with, not a straight line through it
    const curveOffset = 8 + (corridor.id % 3) * 3;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const dx = x2 - x1, dy = y2 - y1;
    const clen = Math.sqrt(dx * dx + dy * dy) || 1;
    const cx = mx + (-dy / clen) * curveOffset, cy = my + (dx / clen) * curveOffset;
    const durationMs = 1800; // compressed demo time per segment
    const start = performance.now();
    log.innerHTML += `<div>&#128666; Departing ${seg.place_from} toward ${seg.place_to} via ${corridor.name}...</div>`;

    function step(now) {
      const t = Math.min(1, (now - start) / durationMs);
      // quadratic bezier point at parameter t
      const bx = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2;
      const by = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2;
      marker.setAttribute('cx', bx);
      marker.setAttribute('cy', by);
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        log.innerHTML += `<div>&#9989; Reached ${seg.place_to}.</div>`;
        segIdx += 1;
        animateSegment();
      }
    }
    requestAnimationFrame(step);
  }
  animateSegment();
}

// ---- launch external turn-by-turn app ----
async function launchPythonRoutingApp() {
  const btn = document.getElementById('launchRouteBtn');
  const status = document.getElementById('routeLaunchStatus');
  btn.disabled = true;
  status.textContent = 'Launching...';
  status.className = 'status-msg';
  try {
    const res = await fetch('/run-routing-script', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      status.textContent = data.message;
      status.className = 'status-msg ok';
    } else {
      status.textContent = data.message || 'Failed to launch.';
      status.className = 'status-msg err';
    }
  } catch (e) {
    status.textContent = 'Could not reach the server.';
    status.className = 'status-msg err';
  } finally {
    btn.disabled = false;
  }
}

populateRouteSelectors();

// ---- GPS capture ----
function getGPSLocation() {
  if (!navigator.geolocation) {
    alert('Geolocation is not supported by this browser.');
    return;
  }
  navigator.geolocation.getCurrentPosition(
    pos => {
      const { latitude, longitude } = pos.coords;
      document.getElementById('gpsCoords').value = `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
    },
    err => alert('Could not get location: ' + err.message)
  );
}

// ---- incident report submit ----
async function handleReportSubmit(event) {
  event.preventDefault();
  const status = document.getElementById('reportStatus');
  const formData = new FormData();
  formData.append('incident_type', document.getElementById('incidentType').value);
  formData.append('gps', document.getElementById('gpsCoords').value);
  const photoInput = document.getElementById('disasterImage');
  if (photoInput.files[0]) formData.append('photo', photoInput.files[0]);

  status.textContent = 'Submitting...';
  status.className = 'status-msg';
  try {
    const res = await fetch('/api/incident-report', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      status.textContent = 'Report submitted. Thank you.';
      status.className = 'status-msg ok';
      event.target.reset();
      document.getElementById('gpsCoords').value = '';
    } else {
      status.textContent = data.message || 'Submission failed.';
      status.className = 'status-msg err';
    }
  } catch (e) {
    status.textContent = 'Could not reach the server.';
    status.className = 'status-msg err';
  }
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# SETUNER — GIS command centre (pure-SVG map, no Leaflet/CDN/tiles)
# ---------------------------------------------------------------------------
SETUNER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SetuNER — Command Centre (Prototype)</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #1c1c1c; }

header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 18px; background: #1F3864; color: #fff;
}
header h1 { font-size: 18px; margin: 0; }
header h1 span { font-weight: 400; font-size: 13px; opacity: .85; margin-left: 10px; }

.badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.badge.online { background: #2e7d32; }
.badge.offline { background: #c62828; }

.layout { display: flex; height: calc(100vh - 46px); }
.sidebar { width: 340px; overflow-y: auto; padding: 14px; border-right: 1px solid #ddd; background: #fafafa; }
.sidebar section { margin-bottom: 18px; }
.sidebar h2 { font-size: 14px; text-transform: uppercase; color: #555; margin: 0 0 8px; }
#map { flex: 1; }

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
button { cursor: pointer; border: 1px solid #ccc; background: #fff; border-radius: 6px; padding: 6px 10px; font-size: 13px; }
button.primary { background: #1F3864; color: #fff; border-color: #1F3864; }
button.full { width: 100%; margin-top: 6px; }
button:disabled { opacity: 0.5; cursor: not-allowed; }

.dev-banner { background: #fff3cd; border: 1px solid #ffe08a; color: #7a5b00; font-size: 11px; padding: 6px 8px; border-radius: 6px; margin-bottom: 8px; }

.corridor-card { border: 1px solid #e2e2e2; border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; background: #fff; }
.corridor-card.spof-card { border-color: #f3b4b4; background: #fff8f8; }
.corridor-card .name { font-weight: 600; font-size: 13px; }
.corridor-card .meta { font-size: 12px; color: #666; }
.corridor-card .score-breakdown { margin-top: 6px; font-size: 10px; color: #888; font-family: monospace; }
.corridor-card .ai-line { margin-top: 4px; font-size: 10px; color: #6b21a8; }
.corridor-card .alt-line { margin-top: 6px; font-size: 11px; color: #1d4ed8; }
.corridor-card .alt-line.muted { color: #9aa9b8; font-style: italic; }
.corridor-card .alt-line.spof { color: #c62828; font-weight: 700; }
.score-pill { float: right; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 700; color: #fff; }
.level-low { background: #2e7d32; }
.level-medium { background: #ef6c00; }
.level-high { background: #c62828; }

.alert-card { border-left: 4px solid #c62828; background: #fff3f3; padding: 8px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; }
.alert-card .alt { margin-top: 4px; font-size: 12px; color: #2e7d32; }

.muted { color: #888; font-size: 13px; }
.small { font-size: 11px; }

#map { position: relative; background: #eaf1f8; }
#map svg { width: 100%; height: 100%; display: block; }
.corridor-line { cursor: pointer; transition: stroke-width .15s; }
.corridor-line:hover { stroke-width: 9; }
.place-label { font-size: 12px; font-weight: 700; fill: #0f172a; }
.place-dot { fill: #ffffff; }
.hospital-dot { fill: #fff; stroke: #c62828; stroke-width: 2; }
.hospital-plus { stroke: #c62828; stroke-width: 2; }
#mapLegend {
  position: absolute; bottom: 12px; left: 12px; background: #fff; border: 1px solid #ddd;
  border-radius: 8px; padding: 8px 12px; font-size: 12px; display: flex; gap: 14px; flex-wrap: wrap; max-width: 90%;
}
#mapLegend span { display: inline-flex; align-items: center; gap: 5px; }
#mapLegend i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }

.switch-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }

.modal { position: fixed; inset: 0; background: rgba(0,0,0,.4); display: flex; align-items: center; justify-content: center; }
.modal.hidden { display: none; }
.modal-card { background: #fff; padding: 20px; border-radius: 10px; width: 320px; }
.modal-card label { display: block; margin-bottom: 10px; font-size: 13px; }
.modal-card input, .modal-card select, .modal-card textarea {
  width: 100%; padding: 6px; margin-top: 4px; border: 1px solid #ccc; border-radius: 5px; font-size: 13px;
}

.model-box { border: 1px solid #e2e2e2; border-radius: 8px; padding: 10px; background: #fff; font-size: 12px; }
.model-box.ok { border-color: #bcd9c2; background: #f3fbf5; }
.model-box.fallback { border-color: #f3d9a4; background: #fffaf0; }
.model-box .row { display: flex; justify-content: space-between; margin-top: 4px; }
.model-box .note { margin-top: 6px; color: #777; font-size: 10.5px; }

.chart-box { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 10px; }
.chart-box svg { width: 100%; height: auto; display: block; }
.chart-note { font-size: 10px; color: #9aa9b8; margin-top: 4px; }

.demo-panel { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 10px; }
.demo-log { max-height: 140px; overflow-y: auto; font-size: 11px; margin-top: 8px; background: #fff; border: 1px solid #e2e2e2; border-radius: 6px; padding: 6px 8px; }
.demo-log div { padding: 2px 0; border-bottom: 1px dashed #eee; }
</style>
</head>
<body>
  <header>
    <h1>SetuNER <span>Smart Logistics &amp; Accessibility Intelligence — NER</span></h1>
    <div style="display:flex; gap:8px;">
      <div id="feedBadge" class="badge online">Live Feed</div>
      <div id="connStatus" class="badge online">Online</div>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <section>
        <h2>Medicine-Truck Demo</h2>
        <div class="demo-panel">
          <p class="small">Runs the full field-report &rarr; AI risk &rarr; GIS &rarr; reroute &rarr; alert story end-to-end.</p>
          <button id="playDemoBtn" class="primary full" onclick="runDemoScenario()"><i></i>&#9654; Play Medicine-Truck Demo</button>
          <div id="demoLog" class="demo-log"></div>
        </div>
      </section>

      <section>
        <h2>Simulate Weather</h2>
        <div class="dev-banner">&#9888; Dev/Demo only — manual trigger standing in for live IMD/MOSDAC feed. Not present in a production build.</div>
        <div class="btn-row">
          <button data-scenario="clear">Clear</button>
          <button data-scenario="moderate">Moderate Rain</button>
          <button data-scenario="heavy">Heavy Rain</button>
          <button data-scenario="severe">Severe Storm</button>
        </div>
        <label class="switch-row" style="margin-top:10px;">
          <input type="checkbox" id="feedToggle" checked>
          Live Weather Feed
        </label>
        <p id="feedStatusNote" class="muted small"></p>
      </section>

      <section>
        <h2>AI Risk Model</h2>
        <div id="modelBox" class="model-box">Loading model status...</div>
      </section>

      <section>
        <h2>Corridors</h2>
        <div id="corridorList"></div>
      </section>

      <section>
        <label class="switch-row">
          <input type="checkbox" id="emergencyToggle">
          Emergency Logistics Mode
        </label>
      </section>

      <section>
        <h2>Active Alerts</h2>
        <div id="alertList"><p class="muted">No active alerts.</p></div>
      </section>

      <section>
        <h2>Analytics</h2>
        <div class="chart-box"><div id="chartArea"></div></div>
        <p class="chart-note">Rendered natively (no chart library / no CDN) so this panel works fully offline too.</p>
      </section>

      <section>
        <button id="reportBtn" class="primary full">+ Submit Field Report</button>
        <button id="syncBtn" class="full" style="display:none;">Sync Queued Reports (<span id="queueCount">0</span>)</button>
        <p class="small muted" style="margin-top:6px;">Offline reports use store-and-forward sync to this server once connectivity returns — this is not device-to-device mesh networking (that would need Bluetooth/LoRa hardware this prototype doesn't have).</p>
      </section>
    </aside>

    <main id="map"></main>
  </div>

  <!-- Field report modal -->
  <div id="reportModal" class="modal hidden">
    <div class="modal-card">
      <h2>Field Intelligence Report</h2>
      <label>Corridor
        <select id="fCorridor"></select>
      </label>
      <label>Report Type
        <select id="fType">
          <option value="blockage">Road Blockage</option>
          <option value="landslide">Landslide</option>
          <option value="road_condition">Road Condition (poor)</option>
          <option value="clear">Route Clear / Verified Safe</option>
        </select>
      </label>
      <label>Officer Name
        <input id="fName" type="text" placeholder="e.g. R. Sangma">
      </label>
      <label>Notes
        <textarea id="fNotes" placeholder="Optional details..."></textarea>
      </label>
      <div class="btn-row">
        <button id="fSubmit" class="primary">Submit</button>
        <button id="fCancel">Cancel</button>
      </div>
      <p class="muted small">If you're offline, this report is queued on-device and synced automatically once connectivity returns.</p>
    </div>
  </div>

<script>
// ---------------------------------------------------------------------------
// Pure-SVG map — no Leaflet, no map tiles, no CDN. Fully offline-safe.
// ---------------------------------------------------------------------------
const BOUNDS = { latMin: 22.8, latMax: 28.0, lngMin: 87.7, lngMax: 95.5, W: 760, H: 680, M: 46 };

function project(lat, lng) {
  const { latMin, latMax, lngMin, lngMax, W, H, M } = BOUNDS;
  const x = M + ((lng - lngMin) / (lngMax - lngMin)) * (W - 2 * M);
  const y = M + ((latMax - lat) / (latMax - latMin)) * (H - 2 * M);
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

let corridorsData = [];
const QUEUE_KEY = 'setuner_offline_queue';
const levelColor = { low: '#2e7d32', medium: '#ef6c00', high: '#c62828' };

// ---- curved "real road" rendering helpers ----
// Alternates share the exact same endpoints as their primary corridor, so a
// straight line would draw them directly on top of each other. Curving each
// one away from the straight line — in opposite directions, by different
// amounts — makes every alternate visibly its own road, like a real bypass
// looping around and rejoining the highway.
function curvedPathD(x1, y1, x2, y2, offset) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const nx = -dy / len, ny = dx / len;
  const cx = mx + nx * offset, cy = my + ny * offset;
  return { d: `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`, midX: mx + nx * offset * 0.5, midY: my + ny * offset * 0.5 };
}

function buildMapSVG(data) {
  const { W, H } = BOUNDS;
  const parts = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">`);

  // terrain-style background (gradient instead of flat fill)
  parts.push(`<defs>
    <radialGradient id="terrainGrad" cx="35%" cy="25%" r="90%">
      <stop offset="0%" stop-color="#f1f6ec"/>
      <stop offset="55%" stop-color="#e7f0e6"/>
      <stop offset="100%" stop-color="#dcebf0"/>
    </radialGradient>
    <radialGradient id="hillGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#c9d9a8" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#c9d9a8" stop-opacity="0"/>
    </radialGradient>
    <filter id="roadShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1.4" stdDeviation="1.3" flood-color="#1c1c1c" flood-opacity="0.25"/>
    </filter>
  </defs>`);
  parts.push(`<rect x="1" y="1" width="${W - 2}" height="${H - 2}" fill="url(#terrainGrad)" stroke="#c9d8e6" rx="14"/>`);

  // soft hill-shading over the NER's actual highland zones (Meghalaya plateau,
  // Naga-Manipur-Mizo hills, Arunachal Himalayan foothills) — schematic relief,
  // not surveyed elevation data, but it reads as real terrain rather than a flat map
  [[25.55, 91.3, 95], [25.7, 94.4, 85], [23.6, 92.9, 80], [27.6, 93.6, 100], [24.1, 93.6, 70]]
    .forEach(([la, ln, r]) => {
      const [hx, hy] = project(la, ln);
      parts.push(`<circle cx="${hx}" cy="${hy}" r="${r}" fill="url(#hillGrad)"/>`);
    });

  for (let i = 1; i < 6; i++) {
    const gx = (W / 6) * i, gy = (H / 6) * i;
    parts.push(`<line x1="${gx}" y1="0" x2="${gx}" y2="${H}" stroke="#dde7ea" stroke-width="1"/>`);
    parts.push(`<line x1="0" y1="${gy}" x2="${W}" y2="${gy}" stroke="#dde7ea" stroke-width="1"/>`);
  }

  // schematic Brahmaputra river (illustrative, not surveyed) for visual context
  const riverPts = [[27.9, 95.3], [26.9, 93.9], [26.4, 92.7], [26.15, 91.75], [26.4, 90.2], [26.1, 88.9]]
    .map(([la, ln]) => project(la, ln));
  const riverD = riverPts.map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`)).join(' ');
  parts.push(`<path d="${riverD}" fill="none" stroke="#a9cbe0" stroke-width="6" stroke-linecap="round" opacity="0.5"/>`);
  parts.push(`<path d="${riverD}" fill="none" stroke="#cfe4ef" stroke-width="2.5" stroke-linecap="round" opacity="0.8"/>`);
  parts.push(`<text x="${riverPts[2][0] - 60}" y="${riverPts[2][1] + 16}" font-size="9" fill="#8fb3c8" font-style="italic">Brahmaputra (schematic)</text>`);

  // small decorative settlement dots (atmosphere only — not corridor endpoints)
  [[26.75, 93.1], [25.85, 92.35], [24.35, 93.1], [26.55, 90.9], [25.05, 92.05], [27.05, 94.4]]
    .forEach(([la, ln]) => {
      const [sx, sy] = project(la, ln);
      parts.push(`<circle cx="${sx}" cy="${sy}" r="1.6" fill="#9fae9f" opacity="0.6"/>`);
    });

  const placeSeen = new Set();
  const alternates = data.corridors.filter(c => c.is_alternate_for);
  const primaries = data.corridors.filter(c => !c.is_alternate_for);
  const altByPrimary = {};
  alternates.forEach(a => { altByPrimary[a.is_alternate_for] = true; });

  function nhCode(name) {
    const m = name.match(/NH ?\\d+[A-Z]?/i);
    return m ? m[0].toUpperCase().replace(' ', '') : null;
  }

  // draw alternates first (so the recommended/primary road always sits on top)
  [...alternates, ...primaries].forEach(c => {
    const [x1, y1] = project(c.from[0], c.from[1]);
    const [x2, y2] = project(c.to[0], c.to[1]);
    const color = levelColor[c.level];
    const isAlt = !!c.is_alternate_for;
    // primaries curve gently one way; alternates curve further the other way,
    // so a primary+alternate pair visibly splits into two distinct roads
    const offset = isAlt ? -26 : (10 + (c.id % 3) * 4);
    const { d, midX, midY } = curvedPathD(x1, y1, x2, y2, offset);
    const width = isAlt ? 4 : 8;
    const dash = isAlt ? 'stroke-dasharray="2,7"' : '';

    // highway casing (light outline) behind the colored road, like Google Maps
    parts.push(`<path d="${d}" fill="none" stroke="#ffffff" stroke-width="${width + 4}" stroke-linecap="round" opacity="0.9" filter="url(#roadShadow)"/>`);
    parts.push(
      `<path class="corridor-line" data-cid="${c.id}" d="${d}" fill="none" stroke="${color}" stroke-width="${width}" ` +
      `stroke-linecap="round" ${dash} opacity="${isAlt ? 0.85 : 0.95}">` +
      `<title>${c.name} — risk ${c.score} (${c.level}), rainfall ${c.rainfall_mm}mm</title></path>`
    );

    // NH shield badge on primaries that carry a route number
    const code = nhCode(c.name);
    if (!isAlt && code) {
      parts.push(`<g><rect x="${midX - 15}" y="${midY - 8}" width="30" height="16" rx="4" fill="#ffffff" stroke="${color}" stroke-width="1.4"/>` +
        `<text x="${midX}" y="${midY + 4}" font-size="9" font-weight="700" text-anchor="middle" fill="${color}">${code}</text></g>`);
    }

    [[c.from, c.from_place], [c.to, c.to_place]].forEach(([coord, label]) => {
      const cleanLabel = label.replace(' (Alt)', '');
      const key = coord.join(',');
      if (placeSeen.has(key)) return;
      placeSeen.add(key);
      const [px, py] = project(coord[0], coord[1]);
      const textW = cleanLabel.length * 6.2 + 6;
      parts.push(`<rect x="${px + 6}" y="${py - 19}" width="${textW}" height="14" rx="3" fill="#ffffff" opacity="0.82"/>`);
      parts.push(`<circle class="place-dot" cx="${px}" cy="${py}" r="6" fill="#ffffff" stroke="#1c1c1c" stroke-width="2.4"/>`);
      parts.push(`<circle cx="${px}" cy="${py}" r="2.4" fill="#1c1c1c"/>`);
      parts.push(`<text class="place-label" x="${px + 9}" y="${py - 8}">${cleanLabel}</text>`);
    });
  });

  data.hospitals.forEach(h => {
    const [x, y] = project(h.lat, h.lng);
    parts.push(
      `<g><circle cx="${x}" cy="${y}" r="9" fill="#ffffff" opacity="0.85"/>` +
      `<circle class="hospital-dot" cx="${x}" cy="${y}" r="8"/>` +
      `<line class="hospital-plus" x1="${x - 4}" y1="${y}" x2="${x + 4}" y2="${y}"/>` +
      `<line class="hospital-plus" x1="${x}" y1="${y - 4}" x2="${x}" y2="${y + 4}"/>` +
      `<title>&#127973; ${h.name}</title></g>`
    );
  });

  // compass rose (top-right)
  const compX = W - 44, compY = 44;
  parts.push(`<g opacity="0.85">
    <circle cx="${compX}" cy="${compY}" r="20" fill="#ffffff" stroke="#c9d8e6"/>
    <path d="M ${compX} ${compY - 14} L ${compX + 5} ${compY} L ${compX} ${compY + 14} L ${compX - 5} ${compY} Z" fill="#1F3864"/>
    <text x="${compX}" y="${compY - 22}" font-size="10" font-weight="700" text-anchor="middle" fill="#1F3864">N</text>
  </g>`);

  // scale bar (bottom-left) — approximate, derived from the projection itself
  const kmPerPxLat = (BOUNDS.latMax - BOUNDS.latMin) * 111 / (H - 2 * BOUNDS.M);
  const barKm = 100;
  const barPx = Math.round(barKm / kmPerPxLat);
  const sbX = 16, sbY = H - 34;
  parts.push(`<g>
    <line x1="${sbX}" y1="${sbY}" x2="${sbX + barPx}" y2="${sbY}" stroke="#333" stroke-width="2"/>
    <line x1="${sbX}" y1="${sbY - 4}" x2="${sbX}" y2="${sbY + 4}" stroke="#333" stroke-width="2"/>
    <line x1="${sbX + barPx}" y1="${sbY - 4}" x2="${sbX + barPx}" y2="${sbY + 4}" stroke="#333" stroke-width="2"/>
    <text x="${sbX}" y="${sbY + 16}" font-size="10" fill="#555">~${barKm} km</text>
  </g>`);

  parts.push(`<text x="${W - 10}" y="${H - 10}" text-anchor="end" font-size="10" fill="#9aa9b8">Schematic road network — rendered natively, not surveyed cartography</text>`);

  parts.push(`</svg>`);
  return parts.join('');
}

function render(data) {
  corridorsData = data.corridors;

  document.getElementById('map').innerHTML =
    buildMapSVG(data) +
    `<div id="mapLegend">
       <span><i style="background:#2e7d32"></i> Low risk</span>
       <span><i style="background:#ef6c00"></i> Medium risk</span>
       <span><i style="background:#c62828"></i> High risk</span>
       <span>&#9679; solid = primary road &nbsp; &#9866; dotted curve = alternate</span>
       <span>&#127973; = hospital</span>
     </div>`;

  document.querySelectorAll('.corridor-line').forEach(el => {
    el.addEventListener('click', () => openReportModal(parseInt(el.dataset.cid)));
  });

  renderSidebarList(data.corridors);
  populateCorridorSelect(data.corridors.filter(c => !c.is_alternate_for));
  renderAnalyticsChart(data.corridors.filter(c => !c.is_alternate_for));
  loadAlerts();
  loadModelInfo();
}

function renderSidebarList(corridors) {
  const el = document.getElementById('corridorList');
  const primaries = corridors.filter(c => !c.is_alternate_for);
  el.innerHTML = primaries.map(c => {
    const alt = corridors.find(a => a.is_alternate_for === c.id);
    const bd = c.breakdown;
    const breakdownHtml = `<div class="score-breakdown">Terrain ${bd.terrain_risk} + History ${bd.historical_disruptions_weighted} (&times;${bd.historical_disruptions_count}) + Rain ${bd.rainfall_weighted} + Field ${bd.field_boost} = <b>${c.score}</b></div>`;
    const aiHtml = c.ai_level
      ? `<div class="ai-line">&#129504; AI cross-check: ${c.ai_level}${c.ai_confidence !== null ? ' (' + Math.round(c.ai_confidence * 100) + '% conf.)' : ''} <i>[${c.ai_source === 'model' ? 'ML model' : 'rule fallback'}]</i></div>`
      : '';
    let altHtml;
    if (c.is_spof) {
      altHtml = `<div class="alt-line spof">&#9888; Critical — no alternate route mapped</div>`;
    } else if (alt) {
      altHtml = `<div class="alt-line">&#8618; Alternate: <b>${alt.name}</b> &middot; risk ${alt.score} (${alt.level})</div>`;
    } else {
      altHtml = `<div class="alt-line muted">No alternate route mapped for this corridor</div>`;
    }
    return `
      <div class="corridor-card${c.is_spof ? ' spof-card' : ''}" id="ccard-${c.id}">
        <span class="score-pill level-${c.level}">${c.score}</span>
        <div class="name">${c.name}</div>
        <div class="meta">${c.from_place} &rarr; ${c.to_place} &middot; ${c.rainfall_mm}mm rainfall</div>
        ${breakdownHtml}
        ${aiHtml}
        ${altHtml}
      </div>
    `;
  }).join('');
}

function populateCorridorSelect(primaryCorridors) {
  const sel = document.getElementById('fCorridor');
  sel.innerHTML = primaryCorridors.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
}

async function loadAlerts() {
  const res = await fetch('/api/alerts');
  const data = await res.json();
  const el = document.getElementById('alertList');
  if (data.alerts.length === 0) {
    el.innerHTML = '<p class="muted">No active alerts.</p>';
    return;
  }
  el.innerHTML = data.alerts.map(a => `
    <div class="alert-card">
      &#9888;&#65039; <b>${a.name}</b> — risk ${a.score} (${a.level})
      ${a.suggested_alternate ? `<div class="alt">&rarr; Suggested alternate: ${a.suggested_alternate.name} (risk ${a.suggested_alternate.score})</div>` : ''}
    </div>
  `).join('');
}

async function loadModelInfo() {
  const res = await fetch('/api/model-info');
  const data = await res.json();
  const box = document.getElementById('modelBox');
  if (data.available) {
    box.className = 'model-box ok';
    box.innerHTML = `
      <div><b>&#9989; ML model active</b> (RandomForest, ${data.cv_folds}-fold CV)</div>
      <div class="row"><span>Accuracy</span><b>${data.accuracy}</b></div>
      <div class="row"><span>Precision (macro)</span><b>${data.precision_macro}</b></div>
      <div class="row"><span>Recall (macro)</span><b>${data.recall_macro}</b></div>
      <div class="row"><span>Training samples</span><b>${data.training_samples}</b></div>
      <div class="note">${data.note}</div>
    `;
  } else {
    box.className = 'model-box fallback';
    box.innerHTML = `<div><b>&#9888; Running on rule-based fallback</b></div><div class="note">${data.reason || 'scikit-learn unavailable in this environment.'}</div>`;
  }
}

// ---- native SVG analytics (no Chart.js / no CDN, so this stays offline-safe) ----
function renderAnalyticsChart(primaryCorridors) {
  const w = 280, h = 160, pad = 26;
  const maxScore = 100;
  const barW = (w - pad * 2) / primaryCorridors.length;
  const bars = primaryCorridors.map((c, i) => {
    const barH = ((h - pad * 2) * c.score) / maxScore;
    const x = pad + i * barW + 4;
    const y = h - pad - barH;
    const color = levelColor[c.level];
    const label = c.name.split(' ')[0];
    return `<rect x="${x}" y="${y}" width="${barW - 8}" height="${barH}" fill="${color}" rx="2"/>` +
           `<text x="${x + (barW - 8) / 2}" y="${h - pad + 12}" font-size="8" text-anchor="middle" fill="#666">${label}</text>` +
           `<text x="${x + (barW - 8) / 2}" y="${y - 3}" font-size="8" text-anchor="middle" fill="#333">${c.score}</text>`;
  }).join('');
  const svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    <line x1="${pad}" y1="${h - pad}" x2="${w - 4}" y2="${h - pad}" stroke="#ccc"/>
    ${bars}
  </svg>`;
  document.getElementById('chartArea').innerHTML = svg;
}

document.querySelectorAll('[data-scenario]').forEach(btn => {
  btn.addEventListener('click', async () => {
    const res = await fetch('/api/simulate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: btn.dataset.scenario })
    });
    render(await res.json());
  });
});

const modal = document.getElementById('reportModal');
function openReportModal(preselectCorridorId) {
  modal.classList.remove('hidden');
  if (preselectCorridorId) document.getElementById('fCorridor').value = preselectCorridorId;
}
document.getElementById('reportBtn').onclick = () => openReportModal();
document.getElementById('fCancel').onclick = () => modal.classList.add('hidden');

document.getElementById('fSubmit').onclick = async () => {
  const payload = {
    corridor_id: parseInt(document.getElementById('fCorridor').value),
    report_type: document.getElementById('fType').value,
    official_name: document.getElementById('fName').value || 'Field Officer',
    notes: document.getElementById('fNotes').value
  };
  modal.classList.add('hidden');

  if (!navigator.onLine) {
    queueReport(payload);
    return;
  }
  try {
    const res = await fetch('/api/field-report', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    render(await res.json());
  } catch (e) {
    queueReport(payload);
  }
};

// Offline reports use store-and-forward sync (queue locally, push once back
// online) — this is NOT device-to-device mesh networking.
function queueReport(payload) {
  const queue = JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]');
  queue.push(payload);
  localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
  updateQueueUI();
}

async function syncQueue() {
  let queue = JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]');
  if (queue.length === 0) return;
  for (const payload of [...queue]) {
    try {
      const res = await fetch('/api/field-report', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        render(await res.json());
        queue.shift();
        localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
      }
    } catch (e) { break; }
  }
  updateQueueUI();
}

function updateQueueUI() {
  const queue = JSON.parse(localStorage.getItem(QUEUE_KEY) || '[]');
  const syncBtn = document.getElementById('syncBtn');
  document.getElementById('queueCount').textContent = queue.length;
  syncBtn.style.display = queue.length > 0 ? 'block' : 'none';
}
document.getElementById('syncBtn').onclick = syncQueue;

function updateConnStatus() {
  const badge = document.getElementById('connStatus');
  if (navigator.onLine) {
    badge.textContent = 'Online'; badge.className = 'badge online';
    syncQueue();
  } else {
    badge.textContent = 'Offline'; badge.className = 'badge offline';
  }
}
window.addEventListener('online', updateConnStatus);
window.addEventListener('offline', updateConnStatus);

// ---- live weather feed outage simulation (the resilience answer: turning
// this off disables only the weather trigger — map, scores, field reports,
// and alerts all keep working on locally-held data) ----
let lastSyncedAt = new Date();
function setFeedOnline(online) {
  const badge = document.getElementById('feedBadge');
  const note = document.getElementById('feedStatusNote');
  document.querySelectorAll('[data-scenario]').forEach(b => b.disabled = !online);
  if (online) {
    badge.textContent = 'Live Feed'; badge.className = 'badge online';
    note.textContent = '';
    lastSyncedAt = new Date();
  } else {
    badge.textContent = 'Feed Offline'; badge.className = 'badge offline';
    note.textContent = `Weather feed unavailable — showing last synced risk data from ${lastSyncedAt.toLocaleTimeString()}. Map, field reports, and alerts remain fully active.`;
  }
}
document.getElementById('feedToggle').addEventListener('change', e => setFeedOnline(e.target.checked));

// ---- scripted medicine-truck demo: the plan's 7-step story, automated ----
function demoLog(msg) {
  const log = document.getElementById('demoLog');
  const line = document.createElement('div');
  line.textContent = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}
function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runDemoScenario() {
  const btn = document.getElementById('playDemoBtn');
  btn.disabled = true;
  document.getElementById('demoLog').innerHTML = '';
  const DEMO_CORRIDOR_ID = 1; // Guwahati - Shillong (NH6)

  try {
    demoLog('Step 1: Medicine truck dispatched Guwahati -> Shillong on NH6 (low risk).');
    let data = await (await fetch('/api/simulate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({scenario:'clear'}) })).json();
    render(data);
    await wait(1400);

    demoLog('Step 2: Heavy rainfall detected (simulated IMD/MOSDAC feed).');
    data = await (await fetch('/api/simulate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({scenario:'heavy'}) })).json();
    render(data);
    await wait(1400);

    const flagged = data.corridors.find(c => c.id === DEMO_CORRIDOR_ID);
    demoLog(`Step 3: AI risk model flags NH6 as ${flagged.ai_level.toUpperCase()} risk${flagged.ai_confidence !== null ? ' (' + Math.round(flagged.ai_confidence*100) + '% confidence)' : ' (rule-based fallback)'}.`);
    document.getElementById('ccard-' + DEMO_CORRIDOR_ID)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await wait(1400);

    demoLog('Step 4: Field officer confirms a landslide on NH6.');
    data = await (await fetch('/api/field-report', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({corridor_id: DEMO_CORRIDOR_ID, report_type: 'landslide', official_name: 'Demo Field Officer', notes: 'Auto-generated by demo scenario'}) })).json();
    render(data);
    await wait(1400);

    demoLog('Step 5: Route engine recalculates — selecting the lowest-risk alternate.');
    const routeRes = await (await fetch('/api/optimize-route', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({origin: 'Guwahati', destination: 'Shillong'}) })).json();
    if (routeRes.status === 'success') {
      demoLog(`Step 6: Truck automatically rerouted via ${routeRes.route.primary.segments[0].corridor.name} (risk ${routeRes.route.primary.total_risk_score}).`);
    }
    await wait(1400);

    const alerts = await (await fetch('/api/alerts')).json();
    demoLog(`Step 7: ${alerts.alerts.length} active alert(s) now visible on the Command Centre dashboard for SDMA/district staff.`);
    loadAlerts();
  } finally {
    btn.disabled = false;
  }
}

fetch('/api/corridors').then(r => r.json()).then(render);
updateConnStatus();
updateQueueUI();

</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS corridors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            from_place TEXT, to_place TEXT,
            from_lat REAL, from_lng REAL,
            to_lat REAL, to_lng REAL,
            terrain_risk INTEGER DEFAULT 0,
            historical_disruptions INTEGER DEFAULT 0,
            rainfall_mm INTEGER DEFAULT 5,
            field_boost INTEGER DEFAULT 0,
            is_alternate_for INTEGER
        );

        CREATE TABLE IF NOT EXISTS field_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            corridor_id INTEGER,
            official_name TEXT,
            report_type TEXT,
            notes TEXT,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS hospitals (
            id INTEGER PRIMARY KEY,
            name TEXT, lat REAL, lng REAL, nearest_corridor_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_type TEXT,
            gps TEXT,
            photo_path TEXT,
            created_at INTEGER
        );
        """
    )
    if db.execute("SELECT COUNT(*) c FROM corridors").fetchone()["c"] == 0:
        # A denser, fully-connected NER network (stand-in for a real OSRM road
        # graph): every cluster now links to its neighbors, so routing can
        # genuinely span multiple hops, not just a single primary/alternate pair.
        corridors = [
            # id, name, from, to, from_lat, from_lng, to_lat, to_lng, terrain_risk, hist_disr, rainfall, field_boost, alt_for
            (1, "Guwahati - Shillong (NH6)", "Guwahati", "Shillong", 26.1445, 91.7362, 25.5788, 91.8933, 35, 4, 5, 0, None),
            (2, "Guwahati - Shillong via Nongpoh (Alt)", "Guwahati", "Shillong (Alt)", 26.1445, 91.7362, 25.5788, 91.8933, 15, 1, 5, 0, 1),
            (3, "Dimapur - Kohima (NH29)", "Dimapur", "Kohima", 25.9091, 93.7266, 25.6751, 94.1086, 22, 2, 5, 0, None),
            (4, "Dimapur - Kohima via Zubza (Alt)", "Dimapur", "Kohima (Alt)", 25.9091, 93.7266, 25.6751, 94.1086, 18, 1, 5, 0, 3),
            (5, "Agartala - Aizawl (NH8)", "Agartala", "Aizawl", 23.8315, 91.2868, 23.7271, 92.7176, 12, 1, 5, 0, None),
            (6, "Itanagar - Guwahati (NH15/NE-A2)", "Itanagar", "Guwahati", 27.0844, 93.6053, 26.1445, 91.7362, 45, 6, 5, 0, None),
            (7, "Guwahati - Dimapur (NH27)", "Guwahati", "Dimapur", 26.1445, 91.7362, 25.9091, 93.7266, 25, 3, 5, 0, None),
            (8, "Shillong - Silchar (NH6 ext.)", "Shillong", "Silchar", 25.5788, 91.8933, 24.8333, 92.7789, 28, 3, 5, 0, None),
            (9, "Silchar - Aizawl (NH306)", "Silchar", "Aizawl", 24.8333, 92.7789, 23.7271, 92.7176, 20, 2, 5, 0, None),
            (10, "Kohima - Imphal (NH2)", "Kohima", "Imphal", 25.6751, 94.1086, 24.8170, 93.9368, 30, 3, 5, 0, None),
            (11, "Imphal - Aizawl (NH102B)", "Imphal", "Aizawl", 24.8170, 93.9368, 23.7271, 92.7176, 33, 4, 5, 0, None),
            # New primaries: two genuine real-world single points of failure, left
            # WITHOUT a mapped alternate on purpose (not an oversight — the
            # Siliguri Corridor really is India's actual chokepoint into the
            # entire NER, and Tura's link is presented as "no alternate mapped
            # yet" to keep one ordinary SPOF case alongside the famous one).
            (12, "Guwahati - Siliguri Corridor / 'Chicken's Neck' (NH27)", "Guwahati", "Siliguri", 26.1445, 91.7362, 26.7271, 88.3953, 40, 5, 5, 0, None),
            (13, "Guwahati - Tura (NH127B)", "Guwahati", "Tura", 26.1445, 91.7362, 25.5138, 90.2172, 20, 2, 5, 0, None),
            # Alternates added for every corridor that previously had none, so
            # the network reflects real redundant highway alignments, not just
            # a single road per city-pair.
            (14, "Agartala - Aizawl via Kamalpur (Alt)", "Agartala", "Aizawl (Alt)", 23.8315, 91.2868, 23.7271, 92.7176, 9, 1, 5, 0, 5),
            (15, "Itanagar - Guwahati via North Bank Highway (Alt)", "Itanagar", "Guwahati (Alt)", 27.0844, 93.6053, 26.1445, 91.7362, 30, 3, 5, 0, 6),
            (16, "Guwahati - Dimapur via Nagaon (Alt)", "Guwahati", "Dimapur (Alt)", 26.1445, 91.7362, 25.9091, 93.7266, 17, 2, 5, 0, 7),
            (17, "Shillong - Silchar via Jowai (Alt)", "Shillong", "Silchar (Alt)", 25.5788, 91.8933, 24.8333, 92.7789, 20, 2, 5, 0, 8),
            (18, "Silchar - Aizawl via Vairengte (Alt)", "Silchar", "Aizawl (Alt)", 24.8333, 92.7789, 23.7271, 92.7176, 14, 1, 5, 0, 9),
            (19, "Kohima - Imphal via Ukhrul (Alt)", "Kohima", "Imphal (Alt)", 25.6751, 94.1086, 24.8170, 93.9368, 22, 2, 5, 0, 10),
            (20, "Imphal - Aizawl via Churachandpur (Alt)", "Imphal", "Aizawl (Alt)", 24.8170, 93.9368, 23.7271, 92.7176, 24, 3, 5, 0, 11),
            # A further round of new routes, adding real NER towns/highways
            (21, "Guwahati - Dibrugarh (NH37)", "Guwahati", "Dibrugarh", 26.1445, 91.7362, 27.4728, 94.9120, 20, 2, 5, 0, None),
            (22, "Guwahati - Dibrugarh via Jorhat (Alt)", "Guwahati", "Dibrugarh (Alt)", 26.1445, 91.7362, 27.4728, 94.9120, 14, 1, 5, 0, 21),
            (23, "Silchar - Agartala (NH8)", "Silchar", "Agartala", 24.8333, 92.7789, 23.8315, 91.2868, 26, 3, 5, 0, None),
            (24, "Silchar - Agartala via Kailashahar (Alt)", "Silchar", "Agartala (Alt)", 24.8333, 92.7789, 23.8315, 91.2868, 19, 2, 5, 0, 23),
            # Deliberately no alternate: Meghalaya's hill state highways genuinely
            # often have only one all-weather alignment between towns.
            (25, "Shillong - Tura (Meghalaya SH)", "Shillong", "Tura", 25.5788, 91.8933, 25.5138, 90.2172, 32, 3, 5, 0, None),
        ]
        db.executemany(
            "INSERT INTO corridors (id,name,from_place,to_place,from_lat,from_lng,to_lat,to_lng,terrain_risk,historical_disruptions,rainfall_mm,field_boost,is_alternate_for) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            corridors,
        )
        hospitals = [
            (1, "Guwahati Medical College", 26.1584, 91.6408, 1),
            (2, "Civil Hospital Shillong", 25.5744, 91.8789, 1),
            (3, "Naga Hospital Kohima", 25.6701, 94.1077, 3),
            (4, "Silchar Medical College Hospital", 24.8500, 92.7900, 8),
            (5, "RIMS Imphal", 24.8000, 93.9300, 10),
            (6, "North Bengal Medical College, Siliguri", 26.7500, 88.4300, 12),
            (7, "Tura Civil Hospital", 25.5150, 90.2200, 13),
            (8, "Assam Medical College, Dibrugarh", 27.4780, 94.9050, 21),
            (9, "Agartala Government Medical College", 23.8400, 91.2700, 5),
        ]
        db.executemany("INSERT INTO hospitals (id,name,lat,lng,nearest_corridor_id) VALUES (?,?,?,?,?)", hospitals)
        db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Rule-based risk engine (the hard-safety-rule fallback the ML layer defers to)
# ---------------------------------------------------------------------------
def compute_score(row):
    score = (
        row["terrain_risk"]
        + row["historical_disruptions"] * 3
        + row["rainfall_mm"] * 0.4
        + row["field_boost"]
    )
    return max(0, min(100, round(score)))


def risk_level(score):
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def corridor_to_dict(row):
    score = compute_score(row)
    ai_level, ai_confidence, ai_source = ml_predict_level(
        row["terrain_risk"], row["historical_disruptions"], row["rainfall_mm"]
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "from_place": row["from_place"],
        "to_place": row["to_place"],
        "from": [row["from_lat"], row["from_lng"]],
        "to": [row["to_lat"], row["to_lng"]],
        "rainfall_mm": row["rainfall_mm"],
        "score": score,
        "level": risk_level(score),
        "is_alternate_for": row["is_alternate_for"],
        # Explainable-risk breakdown — the score is a fusion of four signals
        # this system owns, not a number pulled from a single external API.
        "breakdown": {
            "terrain_risk": row["terrain_risk"],
            "historical_disruptions_count": row["historical_disruptions"],
            "historical_disruptions_weighted": row["historical_disruptions"] * 3,
            "rainfall_mm": row["rainfall_mm"],
            "rainfall_weighted": round(row["rainfall_mm"] * 0.4, 1),
            "field_boost": row["field_boost"],
        },
        # ML cross-check: independent of the rule score above. Falls back to
        # the same thresholds automatically when the model is unavailable or
        # unsure, per the "hard safety rules" guardrail.
        "ai_level": ai_level,
        "ai_confidence": ai_confidence,
        "ai_source": ai_source,
    }


# ---------------------------------------------------------------------------
# Route optimizer — lowest cumulative *risk* path across the corridor graph
# (this is our stand-in for OSRM, since this environment can't reach the
# live OSRM service; the graph itself is now dense enough for genuine
# multi-hop routing, not just single edge-pairs)
# ---------------------------------------------------------------------------
def normalize_place(name):
    """Alternates are stored with a '(Alt)' suffix on to_place purely for display
    in the risk dashboard's corridor list; physically they land on the same town
    as the primary route, so the routing graph needs them collapsed to one node."""
    return name.replace(" (Alt)", "").strip()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_route_graph():
    db = get_db()
    rows = db.execute("SELECT * FROM corridors").fetchall()
    graph = {}
    for r in rows:
        d = corridor_to_dict(r)
        p_from = normalize_place(r["from_place"])
        p_to = normalize_place(r["to_place"])
        dist_km = round(haversine_km(r["from_lat"], r["from_lng"], r["to_lat"], r["to_lng"]), 1)
        graph.setdefault(p_from, []).append(
            (p_to, d["score"], {"place_from": p_from, "place_to": p_to, "distance_km": dist_km, "corridor": d})
        )
        graph.setdefault(p_to, []).append(
            (p_from, d["score"], {"place_from": p_to, "place_to": p_from, "distance_km": dist_km, "corridor": d})
        )
    return graph


def find_top_routes(origin, destination, k=2, max_hops=6):
    """Returns up to k distinct lowest-risk routes (by distinct sets of
    corridor ids used) so the caller can show a recommended route plus a
    genuine alternate rather than only the single best path."""
    origin = normalize_place(origin)
    destination = normalize_place(destination)
    graph = build_route_graph()
    if origin not in graph or destination not in graph:
        return []

    all_paths = []

    def dfs(node, visited, path_edges, risk_sum, dist_sum):
        if len(path_edges) > max_hops:
            return
        if node == destination and path_edges:
            all_paths.append({
                "segments": list(path_edges),
                "total_risk_score": risk_sum,
                "total_distance_km": round(dist_sum, 1),
            })
            return
        for neighbor, weight, edge in graph.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path_edges.append(edge)
            dfs(neighbor, visited, path_edges, risk_sum + weight, dist_sum + edge["distance_km"])
            path_edges.pop()
            visited.remove(neighbor)

    dfs(origin, {origin}, [], 0, 0)
    all_paths.sort(key=lambda p: p["total_risk_score"])

    seen_edge_sets = set()
    unique_routes = []
    for p in all_paths:
        key = tuple(sorted(s["corridor"]["id"] for s in p["segments"]))
        if key in seen_edge_sets:
            continue
        seen_edge_sets.add(key)
        unique_routes.append(p)
        if len(unique_routes) >= k:
            break
    return unique_routes


def get_place_coords(place_name):
    db = get_db()
    place_name = normalize_place(place_name)
    rows = db.execute(
        "SELECT from_place, from_lat, from_lng, to_place, to_lat, to_lng FROM corridors"
    ).fetchall()
    for r in rows:
        if normalize_place(r["from_place"]) == place_name:
            return (r["from_lat"], r["from_lng"])
        if normalize_place(r["to_place"]) == place_name:
            return (r["to_lat"], r["to_lng"])
    return None


def build_spof_advisory(primary_route, destination):
    """When a route has no alternate, don't just go silent — surface the
    nearest emergency rally point and a historical clearance-time estimate."""
    db = get_db()
    dest_coords = get_place_coords(destination)
    hospitals = [dict(h) for h in db.execute("SELECT * FROM hospitals").fetchall()]

    nearest_rally_point = None
    if dest_coords and hospitals:
        best = min(
            hospitals,
            key=lambda h: haversine_km(dest_coords[0], dest_coords[1], h["lat"], h["lng"]),
        )
        nearest_rally_point = {
            "name": best["name"],
            "distance_km": round(
                haversine_km(dest_coords[0], dest_coords[1], best["lat"], best["lng"]), 1
            ),
        }

    total_disruptions = sum(
        s["corridor"]["breakdown"]["historical_disruptions_count"] for s in primary_route["segments"]
    )
    estimated_clearance_hours = round(4 + total_disruptions * 3, 1)

    return {
        "message": "No mapped alternate for this route — this corridor is a single point of failure.",
        "nearest_rally_point": nearest_rally_point,
        "estimated_clearance_hours": estimated_clearance_hours,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def portal():
    return Response(PORTAL_HTML, mimetype="text/html")


@app.route("/setuner")
def setuner_dashboard():
    return Response(SETUNER_HTML, mimetype="text/html")


@app.route("/api/corridors")
def api_corridors():
    db = get_db()
    rows = db.execute("SELECT * FROM corridors").fetchall()
    corridors = [corridor_to_dict(r) for r in rows]
    alt_targets = {c["is_alternate_for"] for c in corridors if c["is_alternate_for"]}
    for c in corridors:
        c["is_spof"] = (c["is_alternate_for"] is None) and (c["id"] not in alt_targets)
    hospitals = [dict(h) for h in db.execute("SELECT * FROM hospitals").fetchall()]
    return jsonify({"corridors": corridors, "hospitals": hospitals})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Apply a weather scenario preset to ALL primary corridors (alternates get
    a smaller bump). DEV/DEMO ONLY — this stands in for a live IMD/MOSDAC feed
    and would not exist in a production build (see the review's "God Mode"
    security note: any panel that injects synthetic conditions must be
    dev-flagged and stripped from production, never left reachable)."""
    scenario = request.json.get("scenario", "clear")
    preset = {"clear": 5, "moderate": 40, "heavy": 90, "severe": 150}.get(scenario, 5)
    db = get_db()
    rows = db.execute("SELECT * FROM corridors").fetchall()
    for r in rows:
        factor = 0.5 if r["is_alternate_for"] else 1.0
        db.execute(
            "UPDATE corridors SET rainfall_mm=? WHERE id=?",
            (int(preset * factor), r["id"]),
        )
    db.commit()
    return api_corridors()


@app.route("/api/field-report", methods=["POST"])
def api_field_report():
    data = request.json
    corridor_id = data["corridor_id"]
    report_type = data["report_type"]
    boost_map = {"blockage": 25, "landslide": 30, "road_condition": 15, "clear": -20}
    boost = boost_map.get(report_type, 0)

    db = get_db()
    db.execute(
        "INSERT INTO field_reports (corridor_id, official_name, report_type, notes, created_at) VALUES (?,?,?,?,?)",
        (corridor_id, data.get("official_name", "Field Officer"), report_type, data.get("notes", ""), int(time.time())),
    )
    row = db.execute("SELECT field_boost FROM corridors WHERE id=?", (corridor_id,)).fetchone()
    new_boost = max(0, row["field_boost"] + boost)
    db.execute("UPDATE corridors SET field_boost=? WHERE id=?", (new_boost, corridor_id))
    db.commit()
    return api_corridors()


@app.route("/api/alerts")
def api_alerts():
    db = get_db()
    rows = db.execute("SELECT * FROM corridors WHERE is_alternate_for IS NULL").fetchall()
    alerts = []
    for r in rows:
        d = corridor_to_dict(r)
        if d["level"] in ("high", "medium"):
            alt = db.execute("SELECT * FROM corridors WHERE is_alternate_for=?", (r["id"],)).fetchone()
            d["suggested_alternate"] = corridor_to_dict(alt) if alt else None
            alerts.append(d)
    return jsonify({"alerts": alerts})


@app.route("/api/places")
def api_places():
    db = get_db()
    rows = db.execute("SELECT DISTINCT from_place, to_place FROM corridors").fetchall()
    places = set()
    for r in rows:
        places.add(normalize_place(r["from_place"]))
        places.add(normalize_place(r["to_place"]))
    return jsonify({"places": sorted(places)})


@app.route("/api/optimize-route", methods=["POST"])
def api_optimize_route():
    data = request.json or {}
    origin = data.get("origin", "")
    destination = data.get("destination", "")
    if not origin or not destination:
        return jsonify({"status": "error", "message": "Origin and destination are required."}), 400
    if normalize_place(origin) == normalize_place(destination):
        return jsonify({"status": "error", "message": "Origin and destination must be different."}), 400

    routes = find_top_routes(origin, destination, k=2)
    if not routes:
        return jsonify({
            "status": "error",
            "message": f"No known route between {origin} and {destination} in this demo network.",
        }), 404

    primary = routes[0]
    alternate = routes[1] if len(routes) > 1 else None
    advisory = build_spof_advisory(primary, destination) if not alternate else None

    return jsonify({
        "status": "success",
        "route": {"primary": primary, "alternate": alternate, "advisory": advisory},
    })


@app.route("/api/incident-report", methods=["POST"])
def api_incident_report():
    incident_type = request.form.get("incident_type", "")
    gps = request.form.get("gps", "")
    if not incident_type or not gps:
        return jsonify({"status": "error", "message": "Incident type and GPS coordinates are required."}), 400

    photo = request.files.get("photo")
    photo_path = None
    if photo and photo.filename:
        safe_name = f"{int(time.time())}_{os.path.basename(photo.filename)}"
        photo_path = os.path.join(UPLOAD_DIR, safe_name)
        photo.save(photo_path)

    db = get_db()
    db.execute(
        "INSERT INTO incidents (incident_type, gps, photo_path, created_at) VALUES (?,?,?,?)",
        (incident_type, gps, photo_path, int(time.time())),
    )
    db.commit()
    return jsonify({"status": "success", "message": "Incident report received."})


@app.route("/api/model-info")
def api_model_info():
    return jsonify(MODEL_METRICS)


@app.route("/api/export/corridors")
def api_export_corridors():
    """Read-only interoperability sketch: a plain JSON feed other government
    systems (NDMA/SDMA dashboards, PWD road databases) could poll. This is a
    first sketch of that story, not a finished data-sharing agreement — the
    review correctly flagged that ownership/access-control model as unwritten."""
    db = get_db()
    rows = db.execute("SELECT * FROM corridors WHERE is_alternate_for IS NULL").fetchall()
    return jsonify({
        "source": "SetuNER prototype — not an authoritative feed",
        "generated_at": int(time.time()),
        "corridors": [corridor_to_dict(r) for r in rows],
    })


@app.route("/run-routing-script", methods=["POST"])
def run_routing_script():
    """Launches the standalone route-navigation prototype as a separate local
    process. The script name is fixed in code (not taken from the request),
    so there's no path-injection risk from the client — but there is also no
    auth check, so anyone who can reach this endpoint can spawn the process.
    Acceptable for a localhost-only demo; add auth before deploying beyond
    127.0.0.1."""
    try:
        subprocess.Popen([sys.executable, ROUTING_SCRIPT])
        return jsonify({"status": "success", "message": "Routing app launched successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# One-time startup work (DB init + ML training) needs to run whether the
# app is launched directly (`python app.py`) or imported by a WSGI server
# like gunicorn (`gunicorn app:app`), since gunicorn never executes the
# __main__ block below.
init_db()
train_ml_model()

if __name__ == "__main__":
    # Hosting platforms (Render, Railway, etc.) assign a port at runtime via
    # the PORT env var and require binding to 0.0.0.0, not 127.0.0.1 — so we
    # use PORT if it's set, and fall back to localhost-only for local dev.
    PORT = int(os.environ.get("PORT", 5000))
    HOST = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    print("\n" + "=" * 60)
    print("  SetuNER is starting...")
    if ML_AVAILABLE and MODEL_METRICS.get("available"):
        print(f"  ML risk model: ACTIVE (accuracy {MODEL_METRICS['accuracy']}, "
              f"precision {MODEL_METRICS['precision_macro']}, recall {MODEL_METRICS['recall_macro']})")
    else:
        print(f"  ML risk model: fallback mode — {MODEL_METRICS.get('reason', 'unavailable')}")
    print(f"  >>>  http://{HOST}:{PORT}  <<<")
    print("=" * 60 + "\n")
    try:
        print("  Server ready.\n")
        app.run(debug=False, use_reloader=False, host=HOST, port=PORT)
    except OSError:
        FALLBACK = 5050
        print(f"\n  Port {PORT} was busy — starting on a fallback port instead.")
        app.run(debug=False, use_reloader=False, host=HOST, port=FALLBACK)
