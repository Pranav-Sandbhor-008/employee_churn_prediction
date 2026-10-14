from flask import Flask, render_template_string, request
import os
import pickle
import warnings
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knn.pkl")

FEATURES = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]

# This pickle should only be loaded if you trust its source.
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

model_feature_count = getattr(model, "n_features_in_", len(FEATURES))
if model_feature_count != len(FEATURES):
    raise ValueError(
        f"The model expects {model_feature_count} features, but this app provides {len(FEATURES)}."
    )

model_feature_names = list(getattr(model, "feature_names_in_", []))
if model_feature_names and model_feature_names != FEATURES:
    raise ValueError(
        "Feature order mismatch. Model expects: " + ", ".join(model_feature_names)
    )

# These are the observed numeric category codes in the fitted model.
# Replace the display labels with your dataset's original category names
# if you have them. Numeric codes must match the training-time encoding.
CATEGORY_OPTIONS = {
    "gender": [(0, "Category 0"), (1, "Category 1"), (2, "Category 2")],
    "course": [(i, f"Course code {i}") for i in range(7)],
    "internet_access": [(0, "No (0)"), (1, "Yes (1)")],
    "sleep_quality": [(0, "Level 0"), (1, "Level 1"), (2, "Level 2")],
    "study_method": [(i, f"Method code {i}") for i in range(5)],
    "facility_rating": [(0, "Rating 0"), (1, "Rating 1"), (2, "Rating 2")],
    "exam_difficulty": [(0, "Level 0"), (1, "Level 1"), (2, "Level 2")],
}

PAGE = r"""
<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Predict an estimated exam score using a trained K-nearest neighbors regression model.">
  <title>ScoreSense | Exam Score Predictor</title>
  <script>
    (() => {
      try {
        const saved = localStorage.getItem("scoresense-theme");
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");
      } catch (_) {}
    })();
  </script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fc; --surface: rgba(255,255,255,.88); --surface-solid: #fff;
      --text: #172033; --muted: #68738a; --border: #e3e8f2;
      --primary: #5855e7; --primary-hover: #4643d1; --accent: #8b5cf6;
      --soft: #efefff; --input: #fff; --shadow: 0 22px 60px rgba(27,39,80,.09);
      --success: #087f5b; --danger: #be3455;
    }
    html[data-theme="dark"] {
      color-scheme: dark;
      --bg: #0b1020; --surface: rgba(20,27,47,.9); --surface-solid: #151c31;
      --text: #eef2ff; --muted: #a4afc5; --border: #2b3550;
      --primary: #9997ff; --primary-hover: #b1adff; --accent: #c1a4ff;
      --soft: #292746; --input: #11182b; --shadow: 0 22px 60px rgba(0,0,0,.28);
      --success: #6ee7b7; --danger: #fda4af;
    }
    @media (prefers-color-scheme: dark) {
      html:not([data-theme="light"]) { color-scheme: dark; }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; color: var(--text);
      background:
        radial-gradient(ellipse at 8% 0%, rgba(88,85,231,.15), transparent 32rem),
        radial-gradient(ellipse at 96% 8%, rgba(139,92,246,.12), transparent 28rem), var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5; transition: background .2s, color .2s;
    }
    .shell { width: min(1120px, calc(100% - 36px)); margin: 0 auto; padding: 26px 0 38px; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:30px; }
    .brand { display:flex; align-items:center; gap:12px; font-weight:850; letter-spacing:-.04em; font-size:1.1rem; }
    .logo { display:grid; place-items:center; width:44px; height:44px; border-radius:15px; color:#fff; background:linear-gradient(135deg,#5855e7,#a855f7); box-shadow:0 8px 22px #5855e733; font-size:1rem; }
    .brand small { display:block; color:var(--muted); font-size:.67rem; letter-spacing:.14em; text-transform:uppercase; font-weight:750; margin-top:1px; }
    .top-actions { display:flex; align-items:center; gap:10px; }
    .status { display:flex; align-items:center; gap:8px; border:1px solid var(--border); background:var(--surface); border-radius:999px; padding:9px 12px; color:var(--muted); font-size:.78rem; }
    .dot { width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow:0 0 0 4px #22c5521c; }
    .theme-toggle { display:flex; align-items:center; gap:7px; border:1px solid var(--border); background:var(--surface); color:var(--text); border-radius:999px; padding:9px 12px; font:inherit; font-size:.8rem; font-weight:700; cursor:pointer; }
    .hero { display:grid; grid-template-columns:1.1fr .9fr; gap:24px; align-items:stretch; margin-bottom:24px; }
    .hero-copy { padding:24px 4px 18px; }
    .eyebrow { display:inline-flex; align-items:center; gap:8px; color:var(--primary); font-size:.75rem; text-transform:uppercase; letter-spacing:.15em; font-weight:850; }
    h1 { font-size:clamp(2.25rem, 5vw, 3.6rem); letter-spacing:-.065em; line-height:1.04; margin:15px 0 18px; max-width:650px; }
    .gradient { background:linear-gradient(100deg,var(--primary),var(--accent)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy p { color:var(--muted); max-width:600px; font-size:1rem; margin:0; }
    .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
    .chip { padding:7px 10px; border:1px solid var(--border); border-radius:999px; font-size:.74rem; color:var(--muted); background:var(--surface); }
    .hero-card, .panel { background:var(--surface); border:1px solid var(--border); border-radius:24px; box-shadow:var(--shadow); backdrop-filter:blur(16px); }
    .hero-card { padding:24px; display:flex; flex-direction:column; justify-content:space-between; min-height:225px; overflow:hidden; position:relative; }
    .hero-card:after { content:""; position:absolute; width:190px; height:190px; border-radius:50%; right:-70px; top:-85px; background:linear-gradient(135deg,#7778f533,#b69aff25); pointer-events:none; }
    .mini-label { color:var(--muted); font-size:.8rem; font-weight:700; }
    .hero-stat { font-size:2rem; font-weight:900; letter-spacing:-.05em; margin:8px 0 2px; }
    .stat-foot { color:var(--muted); font-size:.85rem; }
    .stat-icon { display:grid; place-items:center; width:45px; height:45px; border-radius:14px; background:var(--soft); color:var(--primary); font-size:1.35rem; margin-bottom:16px; }
    .panel { padding:clamp(20px, 4vw, 34px); }
    .section-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:26px; }
    h2 { margin:0 0 6px; font-size:1.35rem; letter-spacing:-.035em; }
    .section-head p { color:var(--muted); margin:0; font-size:.88rem; }
    .step { flex:none; font-size:.72rem; color:var(--primary); background:var(--soft); border-radius:999px; padding:8px 11px; font-weight:850; }
    .group-title { grid-column:1/-1; display:flex; align-items:center; gap:9px; padding-top:3px; color:var(--muted); font-size:.75rem; font-weight:850; letter-spacing:.12em; text-transform:uppercase; }
    .group-title:after { content:""; height:1px; background:var(--border); flex:1; }
    .form-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:19px 16px; }
    .field { min-width:0; }
    label { display:block; font-size:.82rem; font-weight:780; margin:0 0 8px; }
    .hint { display:block; font-size:.72rem; color:var(--muted); margin-top:6px; }
    input, select { width:100%; appearance:none; color:var(--text); background:var(--input); border:1px solid var(--border); border-radius:12px; padding:12px 13px; font:inherit; font-size:.9rem; outline:none; transition:border-color .2s, box-shadow .2s; }
    select { background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),linear-gradient(135deg,var(--muted) 50%,transparent 50%); background-position:calc(100% - 17px) 50%,calc(100% - 12px) 50%; background-size:5px 5px,5px 5px; background-repeat:no-repeat; padding-right:32px; }
    input:focus, select:focus { border-color:var(--primary); box-shadow:0 0 0 3px color-mix(in srgb,var(--primary) 20%,transparent); }
    .actions { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-top:28px; padding-top:23px; border-top:1px solid var(--border); }
    .privacy { color:var(--muted); font-size:.76rem; max-width:490px; }
    .button { border:0; border-radius:13px; padding:13px 19px; color:#fff; font:inherit; font-weight:850; cursor:pointer; background:linear-gradient(110deg,#5855e7,#8b5cf6); box-shadow:0 10px 24px #5855e72b; transition:transform .18s,filter .18s; }
    .button:hover { transform:translateY(-1px); filter:brightness(1.07); }
    .result { margin-top:23px; padding:22px; border-radius:18px; border:1px solid var(--border); background:var(--surface-solid); display:flex; gap:16px; align-items:flex-start; }
    .result-icon { width:46px; height:46px; flex:none; display:grid; place-items:center; border-radius:14px; background:var(--soft); color:var(--primary); font-size:1.3rem; }
    .result h3 { margin:0 0 4px; font-size:1.02rem; }
    .result p { margin:0; color:var(--muted); font-size:.88rem; }
    .score { font-size:2.2rem; font-weight:900; letter-spacing:-.055em; color:var(--primary); margin:7px 0; }
    .score small { font-size:.9rem; letter-spacing:0; font-weight:700; color:var(--muted); }
    .meter { height:7px; background:var(--border); border-radius:99px; overflow:hidden; margin-top:12px; max-width:360px; }
    .meter span { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#5855e7,#a855f7); }
    .error { color:var(--danger); background:var(--surface-solid); border:1px solid var(--border); padding:14px 16px; border-radius:12px; margin-top:18px; }
    .notice { margin-top:18px; color:var(--muted); font-size:.75rem; }
    footer { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:.76rem; padding:22px 4px 0; }
    @media (max-width:850px) { .hero { grid-template-columns:1fr; gap:12px; } .hero-card { min-height:180px; } .form-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:560px) { .shell { width:calc(100% - 24px); padding-top:16px; } .topbar { align-items:flex-start; } .status { display:none; } .hero-copy { padding-top:10px; } .form-grid { grid-template-columns:1fr; } .section-head { flex-direction:column; } .actions { align-items:stretch; flex-direction:column; } .button { width:100%; } .panel { border-radius:19px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><div class="logo">SS</div><div>ScoreSense<small>Student performance analytics</small></div></div>
      <div class="top-actions">
        <div class="status"><span class="dot"></span> KNN model ready</div>
        <button class="theme-toggle" id="themeToggle" type="button" aria-label="Toggle color theme"><span id="themeIcon">☾</span><span id="themeLabel">Dark mode</span></button>
      </div>
    </header>

    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">✦ Predictive learning insights</div>
        <h1>Turn study habits into <span class="gradient">score insights.</span></h1>
        <p>Estimate an exam score using a trained K-nearest neighbors regression model. Adjust the student profile and generate a prediction in seconds.</p>
        <div class="chips"><span class="chip">KNN Regression</span><span class="chip">11 model features</span><span class="chip">Instant estimate</span></div>
      </div>
      <aside class="hero-card">
        <div>
          <div class="stat-icon">⌁</div>
          <div class="mini-label">Prediction workspace</div>
          <div class="hero-stat">Student score estimator</div>
          <div class="stat-foot">Study patterns · Attendance · Learning environment</div>
        </div>
        <div class="chips"><span class="chip">Responsive interface</span><span class="chip">Light / dark theme</span></div>
      </aside>
    </section>

    <section class="panel">
      <div class="section-head">
        <div><h2>Student profile</h2><p>Enter the details below to estimate the predicted exam score.</p></div>
        <span class="step">STEP 01 / PROFILE</span>
      </div>
      <form method="post">
        <div class="form-grid">
          <div class="group-title">Student details</div>
          <div class="field"><label for="age">Age</label><input id="age" name="age" type="number" min="17" max="24" step="1" value="{{ values.get('age','20') }}" required><span class="hint">17–24 years, as observed in training data</span></div>
          <div class="field"><label for="gender">Gender category</label><select id="gender" name="gender" required>{% for v, label in category_options.gender %}<option value="{{ v }}" {% if values.get('gender','0') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Encoded category used by the model</span></div>
          <div class="field"><label for="course">Course category</label><select id="course" name="course" required>{% for v, label in category_options.course %}<option value="{{ v }}" {% if values.get('course','0') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Select the matching training-data code</span></div>

          <div class="group-title">Study routine</div>
          <div class="field"><label for="study_hours">Study hours per day</label><input id="study_hours" name="study_hours" type="number" min="0.08" max="7.91" step="0.01" value="{{ values.get('study_hours','4.0') }}" required><span class="hint">Daily study time (hours)</span></div>
          <div class="field"><label for="class_attendance">Class attendance (%)</label><input id="class_attendance" name="class_attendance" type="number" min="40.6" max="99.4" step="0.1" value="{{ values.get('class_attendance','75.0') }}" required><span class="hint">Attendance percentage</span></div>
          <div class="field"><label for="study_method">Study method</label><select id="study_method" name="study_method" required>{% for v, label in category_options.study_method %}<option value="{{ v }}" {% if values.get('study_method','1') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Encoded method category</span></div>

          <div class="group-title">Wellbeing & learning environment</div>
          <div class="field"><label for="internet_access">Internet access</label><select id="internet_access" name="internet_access" required>{% for v, label in category_options.internet_access %}<option value="{{ v }}" {% if values.get('internet_access','1') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Availability of internet access</span></div>
          <div class="field"><label for="sleep_hours">Sleep hours</label><input id="sleep_hours" name="sleep_hours" type="number" min="4.1" max="9.9" step="0.1" value="{{ values.get('sleep_hours','7.0') }}" required><span class="hint">Average sleep per night</span></div>
          <div class="field"><label for="sleep_quality">Sleep quality</label><select id="sleep_quality" name="sleep_quality" required>{% for v, label in category_options.sleep_quality %}<option value="{{ v }}" {% if values.get('sleep_quality','1') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Encoded quality category</span></div>
          <div class="field"><label for="facility_rating">Facility rating</label><select id="facility_rating" name="facility_rating" required>{% for v, label in category_options.facility_rating %}<option value="{{ v }}" {% if values.get('facility_rating','1') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Encoded rating category</span></div>
          <div class="field"><label for="exam_difficulty">Exam difficulty</label><select id="exam_difficulty" name="exam_difficulty" required>{% for v, label in category_options.exam_difficulty %}<option value="{{ v }}" {% if values.get('exam_difficulty','1') == v|string %}selected{% endif %}>{{ label }}</option>{% endfor %}</select><span class="hint">Encoded difficulty category</span></div>
        </div>
        <div class="actions">
          <div class="privacy">This tool provides a machine-learning estimate, not a guaranteed result. Category codes must match the model's training data.</div>
          <button class="button" type="submit">Predict exam score&nbsp; →</button>
        </div>
      </form>

      {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
      {% if prediction is not none %}
      <div class="result" role="status">
        <div class="result-icon">✦</div>
        <div style="flex:1">
          <h3>Estimated exam score</h3>
          <div class="score">{{ '%.2f'|format(prediction) }} <small>/ 100</small></div>
          <p>Prediction generated by your trained KNN regression model.</p>
          <div class="meter" aria-label="Estimated score level"><span style="width:{{ [0, [prediction, 100]|min]|max }}%"></span></div>
        </div>
      </div>
      {% endif %}
      <div class="notice">Input ranges are based on values observed in the saved model. The categorical display names are placeholders because the original category mapping was not included with the PKL file.</div>
    </section>
    <footer><span>© 2026 ScoreSense · Machine learning demo</span><span>Flask + scikit-learn</span></footer>
  </main>
  <script>
    const root = document.documentElement;
    const toggle = document.getElementById("themeToggle");
    const icon = document.getElementById("themeIcon");
    const label = document.getElementById("themeLabel");
    function syncThemeControls() {
      const dark = root.dataset.theme === "dark";
      icon.textContent = dark ? "☀" : "☾";
      label.textContent = dark ? "Light mode" : "Dark mode";
    }
    toggle.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("scoresense-theme", next); } catch (_) {}
      syncThemeControls();
    });
    syncThemeControls();
  </script>
</body>
</html>
"""

DEFAULTS = {
    "age": "20",
    "gender": "0",
    "course": "0",
    "study_hours": "4.0",
    "class_attendance": "75.0",
    "internet_access": "1",
    "sleep_hours": "7.0",
    "sleep_quality": "1",
    "study_method": "1",
    "facility_rating": "1",
    "exam_difficulty": "1",
}

NUMERIC_RANGES = {
    "age": (17, 24),
    "study_hours": (0.08, 7.91),
    "class_attendance": (40.6, 99.4),
    "sleep_hours": (4.1, 9.9),
}

@app.route("/", methods=["GET", "POST"])
def home():
    values = DEFAULTS.copy()
    prediction = None
    error = None

    if request.method == "POST":
        values = {key: request.form.get(key, "").strip() for key in FEATURES}
        try:
            row = {}
            for name in FEATURES:
                if name in CATEGORY_OPTIONS:
                    allowed = {str(code) for code, _ in CATEGORY_OPTIONS[name]}
                    if values[name] not in allowed:
                        raise ValueError(f"Choose a valid option for {name.replace('_', ' ')}.")
                    row[name] = int(values[name])
                else:
                    row[name] = float(values[name])
                    low, high = NUMERIC_RANGES[name]
                    if not low <= row[name] <= high:
                        raise ValueError(
                            f"{name.replace('_', ' ').capitalize()} must be between {low} and {high}."
                        )

            input_frame = pd.DataFrame([[row[name] for name in FEATURES]], columns=FEATURES)
            prediction = float(np.asarray(model.predict(input_frame)).reshape(-1)[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid numeric result.")
            prediction = max(0.0, min(100.0, prediction))
        except (ValueError, TypeError, KeyError) as exc:
            error = f"Please check your entries. {exc}"
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Check the model file and scikit-learn version in your deployment."

    return render_template_string(
        PAGE,
        values=values,
        prediction=prediction,
        error=error,
        category_options=CATEGORY_OPTIONS,
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
