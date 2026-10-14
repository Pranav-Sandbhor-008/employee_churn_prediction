from flask import Flask, render_template_string, request
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Keep naive.pkl in the same directory as app.py.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "naive.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

EXPECTED_FEATURES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
]

# These options use common LabelEncoder-style mappings for the Employee
# Leave/Retention dataset. Confirm them against your original training code.
EDUCATION_OPTIONS = [("0", "Bachelor's (code 0)"), ("1", "Master's (code 1)"), ("2", "PhD (code 2)")]
CITY_OPTIONS = [("0", "Bangalore (code 0)"), ("1", "New Delhi (code 1)"), ("2", "Pune (code 2)")]
GENDER_OPTIONS = [("0", "Female (code 0)"), ("1", "Male (code 1)")]
BENCHED_OPTIONS = [("0", "No (code 0)"), ("1", "Yes (code 1)")]

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0f172a">
  <title>PeopleLens | Employee Insight</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f4f7fb;
      --surface: rgba(255,255,255,.88);
      --surface-strong: #ffffff;
      --text: #172033;
      --muted: #667085;
      --border: #e4e9f1;
      --primary: #5b5ce2;
      --primary-2: #7c3aed;
      --soft: #efefff;
      --success: #087f5b;
      --danger: #c2415d;
      --shadow: 0 22px 70px rgba(28, 39, 74, .10);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0b1020;
        --surface: rgba(20,27,47,.88);
        --surface-strong: #151c31;
        --text: #eef2ff;
        --muted: #a5afc5;
        --border: #2b3550;
        --primary: #9294ff;
        --primary-2: #b59aff;
        --soft: #292746;
        --success: #6ee7b7;
        --danger: #fda4af;
        --shadow: 0 22px 70px rgba(0,0,0,.28);
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; color: var(--text);
      background:
        radial-gradient(ellipse at 8% 0%, rgba(91,92,226,.16), transparent 34rem),
        radial-gradient(ellipse at 95% 12%, rgba(124,58,237,.11), transparent 28rem),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    .shell { width: min(1120px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0 42px; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:30px; }
    .brand { display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:-.04em; font-size:1.1rem; }
    .logo { width:42px; height:42px; display:grid; place-items:center; color:white; border-radius:14px; background:linear-gradient(135deg,#5b5ce2,#9b5cf6); box-shadow:0 8px 22px #5b5ce23b; }
    .brand small { display:block; font-size:.68rem; color:var(--muted); letter-spacing:.12em; text-transform:uppercase; font-weight:700; margin-top:1px; }
    .status { display:flex; align-items:center; gap:8px; border:1px solid var(--border); background:var(--surface); border-radius:999px; padding:9px 13px; color:var(--muted); font-size:.82rem; }
    .dot { width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow:0 0 0 4px #22c5521c; }
    .hero { display:grid; grid-template-columns:1.1fr .9fr; gap:24px; align-items:stretch; margin-bottom:24px; }
    .hero-copy { padding:26px 4px 22px; }
    .eyebrow { color:var(--primary); font-size:.76rem; text-transform:uppercase; letter-spacing:.16em; font-weight:800; }
    h1 { font-size:clamp(2.15rem, 5vw, 3.55rem); letter-spacing:-.065em; line-height:1.04; margin:14px 0 18px; max-width:680px; }
    .gradient { background:linear-gradient(100deg,var(--primary),var(--primary-2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy p { color:var(--muted); max-width:610px; font-size:1.02rem; margin:0; }
    .hero-card, .panel { background:var(--surface); border:1px solid var(--border); border-radius:24px; box-shadow:var(--shadow); backdrop-filter:blur(16px); }
    .hero-card { padding:24px; display:flex; flex-direction:column; justify-content:space-between; min-height:220px; overflow:hidden; position:relative; }
    .hero-card:after { content:""; position:absolute; width:180px;height:180px;border-radius:50%; right:-60px;top:-70px;background:linear-gradient(135deg,#7778f533,#b69aff25); }
    .mini-label { color:var(--muted); font-size:.82rem; font-weight:650; }
    .big-stat { font-size:2rem; font-weight:850; letter-spacing:-.05em; margin:8px 0 2px; }
    .stat-foot { color:var(--muted); font-size:.86rem; }
    .chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
    .chip { padding:7px 10px; border:1px solid var(--border); border-radius:999px; font-size:.75rem; color:var(--muted); background:var(--surface-strong); }
    .panel { padding:clamp(20px, 4vw, 34px); }
    .section-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:26px; }
    h2 { margin:0 0 6px; font-size:1.35rem; letter-spacing:-.035em; }
    .section-head p { color:var(--muted); margin:0; font-size:.9rem; }
    .step { flex:none; font-size:.75rem; color:var(--primary); background:var(--soft); border-radius:999px; padding:8px 11px; font-weight:800; }
    .form-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:19px 16px; }
    .field { min-width:0; }
    label { display:block; font-size:.83rem; font-weight:750; margin:0 0 8px; }
    .hint { display:block; font-size:.73rem; color:var(--muted); margin-top:6px; }
    input, select { width:100%; appearance:none; color:var(--text); background:var(--surface-strong); border:1px solid var(--border); border-radius:12px; padding:12px 13px; font:inherit; font-size:.9rem; outline:none; transition:border-color .2s, box-shadow .2s; }
    select { background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),linear-gradient(135deg,var(--muted) 50%,transparent 50%); background-position:calc(100% - 17px) 50%,calc(100% - 12px) 50%; background-size:5px 5px,5px 5px; background-repeat:no-repeat; padding-right:32px; }
    input:focus, select:focus { border-color:var(--primary); box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 18%, transparent); }
    .actions { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-top:28px; padding-top:24px; border-top:1px solid var(--border); }
    .privacy { color:var(--muted); font-size:.78rem; max-width:460px; }
    .button { border:0; border-radius:13px; padding:13px 20px; color:white; font:inherit; font-weight:800; cursor:pointer; background:linear-gradient(110deg,#5b5ce2,#8b5cf6); box-shadow:0 10px 24px #5b5ce22b; transition:transform .18s,filter .18s; }
    .button:hover { transform:translateY(-1px); filter:brightness(1.07); }
    .result { margin-top:22px; padding:22px; border-radius:18px; border:1px solid var(--border); background:var(--surface-strong); display:flex; gap:16px; align-items:flex-start; }
    .result-icon { width:44px;height:44px;flex:none;display:grid;place-items:center;border-radius:14px;background:var(--soft);font-size:1.25rem; }
    .result h3 { margin:0 0 5px; font-size:1.08rem; }
    .result p { margin:0; color:var(--muted); font-size:.9rem; }
    .result.stay { border-color:color-mix(in srgb,var(--success) 38%,var(--border)); }
    .result.leave { border-color:color-mix(in srgb,var(--danger) 38%,var(--border)); }
    .error { color:var(--danger); background:var(--surface-strong); border:1px solid var(--border); padding:14px 16px; border-radius:12px; margin-top:18px; }
    .notice { margin-top:18px; color:var(--muted); font-size:.77rem; }
    footer { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:.78rem; padding:22px 4px 0; }
    @media (max-width:900px) { .hero { grid-template-columns:1fr; gap:12px; } .hero-card { min-height:180px; } .form-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:560px) { .shell { width:min(100% - 24px,1120px); padding-top:17px; } .topbar { align-items:flex-start; } .status { font-size:.7rem; padding:8px 10px; } .hero-copy { padding-top:12px; } .form-grid { grid-template-columns:1fr; } .section-head { flex-direction:column; } .actions { align-items:stretch; flex-direction:column; } .button { width:100%; } .panel { border-radius:19px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><div class="logo">PL</div><div>PeopleLens<small>Workforce intelligence</small></div></div>
      <div class="status"><span class="dot"></span> ML model loaded</div>
    </header>

    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">People analytics · Predictive insights</div>
        <h1>Understand your workforce with <span class="gradient">data-driven clarity.</span></h1>
        <p>Explore an employee retention prediction using your trained Naive Bayes model. Enter the employee profile below to generate a model estimate.</p>
        <div class="chip-row"><span class="chip">Gaussian Naive Bayes</span><span class="chip">Instant inference</span><span class="chip">Responsive design</span></div>
      </div>
      <aside class="hero-card">
        <div>
          <div class="mini-label">Prediction workspace</div>
          <div class="big-stat">Employee insight</div>
          <div class="stat-foot">Eight input features · One model output</div>
        </div>
        <div class="chip-row"><span class="chip">Privacy-minded</span><span class="chip">Light / dark theme</span></div>
      </aside>
    </section>

    <section class="panel">
      <div class="section-head">
        <div><h2>Employee profile</h2><p>Complete the fields to estimate the model's predicted class.</p></div>
        <span class="step">STEP 01 / INPUT</span>
      </div>
      <form method="post">
        <div class="form-grid">
          <div class="field"><label for="education">Education</label>
            <select id="education" name="Education" required>
              {% for value, label in education_options %}<option value="{{ value }}" {% if values.get('Education','0') == value %}selected{% endif %}>{{ label }}</option>{% endfor %}
            </select><span class="hint">Encoded category expected by model</span>
          </div>
          <div class="field"><label for="joining_year">Joining year</label>
            <input id="joining_year" name="JoiningYear" type="number" min="2000" max="2035" value="{{ values.get('JoiningYear','2018') }}" required>
            <span class="hint">Year the employee joined</span>
          </div>
          <div class="field"><label for="city">City</label>
            <select id="city" name="City" required>{% for value, label in city_options %}<option value="{{ value }}" {% if values.get('City','0') == value %}selected{% endif %}>{{ label }}</option>{% endfor %}</select>
            <span class="hint">Encoded category expected by model</span>
          </div>
          <div class="field"><label for="payment_tier">Payment tier</label>
            <select id="payment_tier" name="PaymentTier" required>{% for v in [1,2,3] %}<option value="{{ v }}" {% if values.get('PaymentTier','2') == v|string %}selected{% endif %}>Tier {{ v }}</option>{% endfor %}</select>
            <span class="hint">Tier 1, 2 or 3</span>
          </div>
          <div class="field"><label for="age">Age</label>
            <input id="age" name="Age" type="number" min="18" max="100" value="{{ values.get('Age','29') }}" required>
            <span class="hint">Age in years</span>
          </div>
          <div class="field"><label for="gender">Gender</label>
            <select id="gender" name="Gender" required>{% for value, label in gender_options %}<option value="{{ value }}" {% if values.get('Gender','0') == value %}selected{% endif %}>{{ label }}</option>{% endfor %}</select>
            <span class="hint">Encoded category expected by model</span>
          </div>
          <div class="field"><label for="benched">Ever benched?</label>
            <select id="benched" name="EverBenched" required>{% for value, label in benched_options %}<option value="{{ value }}" {% if values.get('EverBenched','0') == value %}selected{% endif %}>{{ label }}</option>{% endfor %}</select>
            <span class="hint">Whether the employee was ever unassigned</span>
          </div>
          <div class="field"><label for="experience">Experience in current domain</label>
            <input id="experience" name="ExperienceInCurrentDomain" type="number" min="0" max="50" value="{{ values.get('ExperienceInCurrentDomain','3') }}" required>
            <span class="hint">Years of domain experience</span>
          </div>
        </div>
        <div class="actions">
          <div class="privacy">Your entries are used to make a prediction by this app. This result is an estimate, not a decision about an employee.</div>
          <button class="button" type="submit">Generate prediction&nbsp; →</button>
        </div>
      </form>

      {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
      {% if prediction is not none %}
        <div class="result {{ 'leave' if prediction == 1 else 'stay' }}" role="status">
          <div class="result-icon">{{ '↗' if prediction == 1 else '✓' }}</div>
          <div>
            <h3>{{ 'Predicted class: 1' if prediction == 1 else 'Predicted class: 0' }}</h3>
            <p>{{ 'The model returned class 1. If your training target is LeaveOrNot with 1 = leave, this indicates a leave prediction.' if prediction == 1 else 'The model returned class 0. If your training target is LeaveOrNot with 0 = stay, this indicates a stay prediction.' }}</p>
            {% if probability is not none %}<p style="margin-top:8px"><strong>Model probability for class {{ prediction }}:</strong> {{ '%.1f'|format(probability * 100) }}%</p>{% endif %}
          </div>
        </div>
      {% endif %}
      <div class="notice">Model compatibility note: categorical values must use the same numeric encoding as your training pipeline. Check the mapping below the code before relying on predictions.</div>
    </section>
    <footer><span>© 2026 PeopleLens · Machine learning demo</span><span>Built with Flask + scikit-learn</span></footer>
  </main>
</body>
</html>
"""

def validate_model():
    feature_names = list(getattr(model, "feature_names_in_", []))
    if feature_names and feature_names != EXPECTED_FEATURES:
        raise ValueError(
            "Model feature names/order do not match this app. "
            f"Expected {EXPECTED_FEATURES}, got {feature_names}"
        )
    feature_count = getattr(model, "n_features_in_", len(EXPECTED_FEATURES))
    if feature_count != len(EXPECTED_FEATURES):
        raise ValueError(f"Model expects {feature_count} features, but app provides {len(EXPECTED_FEATURES)}.")

validate_model()

@app.route("/", methods=["GET", "POST"])
def home():
    values = {
        "Education": "0",
        "JoiningYear": "2018",
        "City": "0",
        "PaymentTier": "2",
        "Age": "29",
        "Gender": "0",
        "EverBenched": "0",
        "ExperienceInCurrentDomain": "3",
    }
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        values = {name: request.form.get(name, "").strip() for name in EXPECTED_FEATURES}
        try:
            numeric = {
                "Education": int(values["Education"]),
                "JoiningYear": int(values["JoiningYear"]),
                "City": int(values["City"]),
                "PaymentTier": int(values["PaymentTier"]),
                "Age": int(values["Age"]),
                "Gender": int(values["Gender"]),
                "EverBenched": int(values["EverBenched"]),
                "ExperienceInCurrentDomain": int(values["ExperienceInCurrentDomain"]),
            }
            if not 2000 <= numeric["JoiningYear"] <= 2035:
                raise ValueError("Joining year must be between 2000 and 2035.")
            if not 18 <= numeric["Age"] <= 100:
                raise ValueError("Age must be between 18 and 100.")
            if numeric["PaymentTier"] not in (1, 2, 3):
                raise ValueError("Payment tier must be 1, 2, or 3.")
            if numeric["Education"] not in (0, 1, 2):
                raise ValueError("Education code must be 0, 1, or 2.")
            if numeric["City"] not in (0, 1, 2):
                raise ValueError("City code must be 0, 1, or 2.")
            if numeric["Gender"] not in (0, 1) or numeric["EverBenched"] not in (0, 1):
                raise ValueError("Gender and EverBenched must use codes 0 or 1.")
            if not 0 <= numeric["ExperienceInCurrentDomain"] <= 50:
                raise ValueError("Domain experience must be between 0 and 50.")

            input_df = pd.DataFrame([numeric], columns=EXPECTED_FEATURES)
            prediction = int(model.predict(input_df)[0])
            if hasattr(model, "predict_proba"):
                probability = float(np.max(model.predict_proba(input_df)[0]))
        except (ValueError, TypeError, KeyError) as exc:
            error = f"Please check your entries. {exc}"
        except Exception:
            app.logger.exception("Prediction failed")
            error = "The model could not generate a prediction. Verify the input encoding and model/library compatibility."

    return render_template_string(
        PAGE,
        values=values,
        prediction=prediction,
        probability=probability,
        error=error,
        education_options=EDUCATION_OPTIONS,
        city_options=CITY_OPTIONS,
        gender_options=GENDER_OPTIONS,
        benched_options=BENCHED_OPTIONS,
    )

if __name__ == "__main__":
    # Render provides PORT at runtime; this also works for local testing.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
