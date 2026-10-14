
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Keep naive.pkl in the same folder as app.py.
MODEL_PATH = BASE_DIR / "naive.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "naive(2).pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Add naive.pkl or naive(2).pkl beside app.py."
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
]

# Verify these encodings against your original training notebook.
EDUCATION = {"0": "Bachelor's", "1": "Master's", "2": "PhD"}
CITIES = {"0": "Bangalore", "1": "New Delhi", "2": "Pune"}
GENDERS = {"0": "Female", "1": "Male"}
BENCHED = {"0": "No", "1": "Yes"}

# Assumption: 0 = Stay, 1 = Leave.
# Change these if your training target used a different mapping.
STAY_CLASS = 0
LEAVE_CLASS = 1

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PeopleIQ | Employee Prediction</title>
<style>
:root {
 --bg:#f3f5fc; --card:#fff; --text:#182033; --muted:#667085;
 --border:#e0e5ef; --field:#f8faff; --primary:#6057df;
 --shadow:0 18px 50px rgba(28,35,75,.10);
}
[data-theme="dark"] {
 --bg:#0d1321; --card:#171f30; --text:#f2f5ff; --muted:#a8b3ca;
 --border:#303b51; --field:#101827; --primary:#a09aff;
 --shadow:0 18px 50px rgba(0,0,0,.25);
 color-scheme:dark;
}
* {box-sizing:border-box}
body {
 margin:0; font-family:Inter,system-ui,Segoe UI,sans-serif;
 background:radial-gradient(ellipse at top left,#7770e51c,transparent 40%),var(--bg);
 color:var(--text); min-height:100vh; transition:.25s;
}
.container {max-width:1100px;width:calc(100% - 32px);margin:auto;padding:28px 0 40px}
header {display:flex;justify-content:space-between;align-items:center;gap:14px}
.brand {display:flex;align-items:center;gap:12px}
.logo {
 background:linear-gradient(135deg,#6258e7,#9870ed);color:white;
 width:46px;height:46px;border-radius:15px;display:grid;place-items:center;
 font-size:23px;font-weight:900;
}
h1 {font-size:19px;margin:0}
.brand p {font-size:12px;color:var(--muted);margin:4px 0 0}
button {cursor:pointer;font:inherit}
.theme {
 border:1px solid var(--border);background:var(--card);color:var(--text);
 padding:10px 14px;border-radius:12px;font-weight:700;
}
.hero {padding:36px 0 25px}
.eyebrow {color:var(--primary);font-size:12px;font-weight:800;letter-spacing:1.5px}
.hero h2 {font-size:clamp(30px,5vw,44px);letter-spacing:-1.2px;margin:10px 0}
.hero p {color:var(--muted);line-height:1.7;max-width:720px}
.layout {display:grid;grid-template-columns:1.35fr .65fr;gap:22px;align-items:start}
.card {background:var(--card);border:1px solid var(--border);border-radius:22px;box-shadow:var(--shadow);overflow:hidden}
.cardhead {padding:22px 24px;border-bottom:1px solid var(--border)}
.cardhead h3 {margin:0 0 7px;font-size:18px}
.cardhead p,.info p {color:var(--muted);font-size:13px;line-height:1.6;margin:0}
form {padding:23px 24px}
.section {font-size:11px;font-weight:850;letter-spacing:1.2px;color:var(--primary);margin:0 0 14px}
.fields {display:grid;grid-template-columns:1fr 1fr;gap:17px;margin-bottom:24px}
label {display:block;font-size:13px;font-weight:750;margin-bottom:8px}
input,select {
 width:100%;padding:12px;border:1px solid var(--border);border-radius:11px;
 background:var(--field);color:var(--text);font:inherit;font-size:14px;outline:none;
}
input:focus,select:focus {border-color:var(--primary);box-shadow:0 0 0 3px #6057df22}
small {display:block;color:var(--muted);font-size:11px;margin-top:6px}
.submit {
 width:100%;border:0;border-radius:12px;padding:15px;color:white;font-weight:850;
 background:linear-gradient(135deg,#6258e7,#8b65e8);box-shadow:0 8px 20px #6258e733;
}
.submit:hover {filter:brightness(1.08);transform:translateY(-1px)}
.side {display:grid;gap:18px}
.info {padding:22px}
.info h3 {font-size:16px;margin:0 0 12px}
.stats {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}
.stat {padding:13px;border:1px solid var(--border);border-radius:12px;background:var(--field)}
.stat strong {display:block;font-size:20px}
.stat span {font-size:11px;color:var(--muted)}
.result {display:flex;gap:15px;padding:19px;margin:0 24px 24px;border-radius:15px}
.result.stay {background:#e8f9ef;border:2px solid #159653;color:#14532d}
.result.leave {background:#fff0f1;border:2px solid #dc3545;color:#842029}
[data-theme="dark"] .result.stay {background:#102d24;color:#8df0b6}
[data-theme="dark"] .result.leave {background:#351b27;color:#ff9aa4}
.result-icon {width:45px;height:45px;flex-shrink:0;border-radius:13px;background:#ffffff80;display:grid;place-items:center;font-size:25px;font-weight:900}
.result h3 {margin:0 0 9px;font-size:20px}
.result p {font-size:13px;line-height:1.55;margin:6px 0}
.badge {display:inline-block;font-size:12px;font-weight:900;letter-spacing:.7px;border-radius:999px;padding:7px 11px;background:#ffffff8c;margin:5px 0}
.error {margin:0 24px 22px;padding:14px;border-radius:12px;background:#fff0f1;color:#a61b29;font-size:13px}
.note {margin-top:16px;background:var(--field);padding:13px;border-radius:12px;font-size:11px;line-height:1.65;color:var(--muted)}
footer {text-align:center;color:var(--muted);font-size:12px;padding-top:25px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.side{grid-template-columns:1fr 1fr}}
@media(max-width:550px){.fields,.side{grid-template-columns:1fr}.container{width:calc(100% - 20px)}form,.cardhead{padding:18px}.result{margin:0 18px 18px}.hero{padding-top:28px}}
</style>
</head>
<body>
<div class="container">
<header>
 <div class="brand">
  <div class="logo">P</div>
  <div><h1>PeopleIQ</h1><p>AI-powered employee retention prediction</p></div>
 </div>
 <button class="theme" id="themeToggle" type="button">☾ Dark mode</button>
</header>

<section class="hero">
 <div class="eyebrow">MACHINE LEARNING · EMPLOYEE ANALYTICS</div>
 <h2>Will your employee<br>stay or leave?</h2>
 <p>Enter the employee's profile and experience details to generate a clear retention prediction using your trained machine learning model.</p>
</section>

<main class="layout">
<section class="card">
 <div class="cardhead">
  <h3>Employee profile</h3>
  <p>Fill in all eight attributes to generate the prediction.</p>
 </div>

 <form method="POST">
  <div class="section">EMPLOYMENT DETAILS</div>
  <div class="fields">
   <div>
    <label for="Education">Education</label>
    <select name="Education" id="Education" required>
    {% for v,label in education.items() %}
    <option value="{{v}}" {% if data.get('Education','0') == v %}selected{% endif %}>{{label}}</option>
    {% endfor %}
    </select>
   </div>
   <div>
    <label for="JoiningYear">Joining year</label>
    <input id="JoiningYear" name="JoiningYear" type="number" min="2000" max="2035" value="{{data.get('JoiningYear','2018')}}" required>
   </div>
   <div>
    <label for="City">City</label>
    <select name="City" id="City" required>
    {% for v,label in cities.items() %}
    <option value="{{v}}" {% if data.get('City','0') == v %}selected{% endif %}>{{label}}</option>
    {% endfor %}
    </select>
   </div>
   <div>
    <label for="PaymentTier">Payment tier</label>
    <select name="PaymentTier" id="PaymentTier" required>
    {% for v,label in [('1','Tier 1'),('2','Tier 2'),('3','Tier 3')] %}
    <option value="{{v}}" {% if data.get('PaymentTier','3') == v %}selected{% endif %}>{{label}}</option>
    {% endfor %}
    </select>
   </div>
  </div>

  <div class="section">PERSONAL & EXPERIENCE DETAILS</div>
  <div class="fields">
   <div>
    <label for="Age">Age</label>
    <input id="Age" name="Age" type="number" min="18" max="100" value="{{data.get('Age','28')}}" required>
   </div>
   <div>
    <label for="Gender">Gender</label>
    <select name="Gender" id="Gender" required>
    {% for v,label in genders.items() %}
    <option value="{{v}}" {% if data.get('Gender','1') == v %}selected{% endif %}>{{label}}</option>
    {% endfor %}
    </select>
   </div>
   <div>
    <label for="EverBenched">Ever benched?</label>
    <select name="EverBenched" id="EverBenched" required>
    {% for v,label in benched.items() %}
    <option value="{{v}}" {% if data.get('EverBenched','0') == v %}selected{% endif %}>{{label}}</option>
    {% endfor %}
    </select>
   </div>
   <div>
    <label for="ExperienceInCurrentDomain">Experience in current domain</label>
    <input id="ExperienceInCurrentDomain" name="ExperienceInCurrentDomain" type="number" min="0" max="60" value="{{data.get('ExperienceInCurrentDomain','3')}}" required>
    <small>Years in the current domain</small>
   </div>
  </div>
  <button class="submit" type="submit">✦ Predict Employee Outcome</button>
 </form>

 {% if error %}
 <div class="error"><strong>Prediction error:</strong> {{error}}</div>
 {% endif %}

 {% if prediction is not none %}
 <div class="result {{'stay' if prediction == stay_class else 'leave'}}" role="status" aria-live="polite">
  <div class="result-icon">{{'✓' if prediction == stay_class else '!'}}</div>
  <div>
   <h3>{{'Employee is Predicted to Stay' if prediction == stay_class else 'Employee is Predicted to Leave'}}</h3>
   <p>{{'The model classifies this employee as likely to remain with the organization.' if prediction == stay_class else 'The model classifies this employee as likely to leave the organization.'}}</p>
   <div class="badge">PREDICTION: {{'STAY' if prediction == stay_class else 'LEAVE'}}</div>
   <p><strong>Predicted class:</strong> {{prediction}}</p>
   {% if leave_probability is not none %}
   <p><strong>Estimated probability of leaving:</strong> {{'%.2f'|format(leave_probability * 100)}}%</p>
   {% endif %}
  </div>
 </div>
 {% endif %}
</section>

<aside class="side">
 <section class="card info">
  <h3>How it works</h3>
  <p>The model evaluates eight employee attributes and returns a predicted class. The result is displayed in simple language for quick interpretation.</p>
  <div class="stats">
   <div class="stat"><strong>08</strong><span>Input attributes</span></div>
   <div class="stat"><strong>ML</strong><span>Model powered</span></div>
  </div>
 </section>
 <section class="card info">
  <h3>Understanding your result</h3>
  <p><strong>STAY</strong> means the model predicts class 0. <strong>LEAVE</strong> means it predicts class 1. The probability is the model's estimate, not a guarantee.</p>
  <div class="note"><strong>Important:</strong> Verify the target labels and categorical encodings against the notebook used to train the model. This tool should support—not replace—fair human review of employment decisions.</div>
 </section>
</aside>
</main>
<footer>PeopleIQ · Employee Retention Prediction · Powered by machine learning</footer>
</div>

<script>
const root = document.documentElement;
const toggle = document.getElementById("themeToggle");

function setTheme(theme) {
 root.setAttribute("data-theme", theme);
 toggle.textContent = theme === "dark" ? "☀ Light mode" : "☾ Dark mode";
 try { localStorage.setItem("peopleiq-theme", theme); } catch(e) {}
}
let theme = "light";
try { theme = localStorage.getItem("peopleiq-theme") || "light"; } catch(e) {}
setTheme(theme);
toggle.addEventListener("click", () => {
 setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
});
</script>
</body>
</html>
"""


def build_input(form):
    values = {
        "Education": int(form["Education"]),
        "JoiningYear": int(form["JoiningYear"]),
        "City": int(form["City"]),
        "PaymentTier": int(form["PaymentTier"]),
        "Age": int(form["Age"]),
        "Gender": int(form["Gender"]),
        "EverBenched": int(form["EverBenched"]),
        "ExperienceInCurrentDomain": int(form["ExperienceInCurrentDomain"]),
    }

    if values["Education"] not in (0, 1, 2):
        raise ValueError("Please select a valid education level.")
    if values["City"] not in (0, 1, 2):
        raise ValueError("Please select a valid city.")
    if values["Gender"] not in (0, 1):
        raise ValueError("Please select a valid gender.")
    if values["EverBenched"] not in (0, 1):
        raise ValueError("Please select a valid benched option.")
    if values["PaymentTier"] not in (1, 2, 3):
        raise ValueError("Payment tier must be 1, 2, or 3.")
    if not 18 <= values["Age"] <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if not 2000 <= values["JoiningYear"] <= 2035:
        raise ValueError("Please enter a valid joining year.")
    if not 0 <= values["ExperienceInCurrentDomain"] <= 60:
        raise ValueError("Experience must be between 0 and 60 years.")

    return pd.DataFrame([values], columns=FEATURES)


def render_page(prediction=None, leave_probability=None, error=None):
    return render_template_string(
        HTML,
        education=EDUCATION,
        cities=CITIES,
        genders=GENDERS,
        benched=BENCHED,
        data=request.form if request.method == "POST" else {},
        prediction=prediction,
        leave_probability=leave_probability,
        error=error,
        stay_class=STAY_CLASS,
    )


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_page()

    try:
        input_data = build_input(request.form)

        # Keep feature order identical to the model's training order.
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None and list(expected_features) != FEATURES:
            raise ValueError(
                "Feature order mismatch. Expected by model: "
                + ", ".join(expected_features)
            )

        prediction = int(model.predict(input_data)[0])

        leave_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if LEAVE_CLASS in classes:
                leave_probability = float(
                    probabilities[classes.index(LEAVE_CLASS)]
                )

        return render_page(
            prediction=prediction,
            leave_probability=leave_probability,
        )

    except (ValueError, TypeError, KeyError) as exc:
        return render_page(error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_page(
            error="Prediction failed. Please check the model and input encodings."
        ), 500


@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
