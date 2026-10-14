"""
Employee Attrition Predictor - Flask app (Render-ready)
Model: Gaussian Naive Bayes (naive.pkl)
Predicts whether an employee is likely to LEAVE or STAY.
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "naive.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact column order the model was trained on
FEATURES = [
    "Education", "JoiningYear", "City", "PaymentTier",
    "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain",
]

# Categorical encodings (alphabetical label-encoding, as used in training)
EDUCATION = {"Bachelors": 0, "Masters": 1, "PHD": 2}
CITY = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}
GENDER = {"Female": 0, "Male": 1}
BENCHED = {"No": 0, "Yes": 1}


def _lookup(mapping, value, field):
    """Accept the text label (case-insensitive) or the numeric code."""
    if isinstance(value, str):
        for key, code in mapping.items():
            if key.lower() == value.strip().lower():
                return code
        if value.strip().isdigit() and int(value) in mapping.values():
            return int(value)
    elif isinstance(value, (int, float)) and int(value) in mapping.values():
        return int(value)
    raise ValueError(f"Invalid {field}: {value!r}. Allowed: {list(mapping)}")


def build_row(data: dict) -> pd.DataFrame:
    """Validate raw input and convert it into a one-row model DataFrame."""
    missing = [k for k in FEATURES if k not in data or data[k] in ("", None)]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    row = {
        "Education": _lookup(EDUCATION, data["Education"], "Education"),
        "JoiningYear": int(data["JoiningYear"]),
        "City": _lookup(CITY, data["City"], "City"),
        "PaymentTier": int(data["PaymentTier"]),
        "Age": int(data["Age"]),
        "Gender": _lookup(GENDER, data["Gender"], "Gender"),
        "EverBenched": _lookup(BENCHED, data["EverBenched"], "EverBenched"),
        "ExperienceInCurrentDomain": int(data["ExperienceInCurrentDomain"]),
    }
    if row["PaymentTier"] not in (1, 2, 3):
        raise ValueError("PaymentTier must be 1, 2 or 3.")
    if not 18 <= row["Age"] <= 70:
        raise ValueError("Age must be between 18 and 70.")
    if not 1990 <= row["JoiningYear"] <= 2100:
        raise ValueError("JoiningYear looks invalid.")
    if row["ExperienceInCurrentDomain"] < 0:
        raise ValueError("ExperienceInCurrentDomain cannot be negative.")
    return pd.DataFrame([row], columns=FEATURES)


def predict(data: dict) -> dict:
    X = build_row(data)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    return {
        "prediction": "Likely to Leave" if pred == 1 else "Likely to Stay",
        "leave_probability": round(float(proba[classes.index(1)]) * 100, 2),
        "stay_probability": round(float(proba[classes.index(0)]) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Employee Attrition Predictor</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f3f4f8;margin:0;padding:24px;color:#1f2937}
  .card{max-width:640px;margin:0 auto;background:#fff;border-radius:14px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
  h1{margin:0 0 4px;font-size:1.5rem}
  p.sub{margin:0 0 20px;color:#6b7280}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  label{font-size:.85rem;font-weight:600;display:block;margin-bottom:4px}
  input,select{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:1rem;background:#fff}
  button{margin-top:20px;width:100%;padding:12px;border:0;border-radius:8px;background:#4f46e5;color:#fff;font-size:1rem;font-weight:600;cursor:pointer}
  button:hover{background:#4338ca}
  .result{margin-top:20px;padding:16px;border-radius:10px;text-align:center}
  .leave{background:#fee2e2;color:#991b1b}
  .stay{background:#dcfce7;color:#166534}
  .error{background:#fef3c7;color:#92400e}
  .result h2{margin:0 0 6px}
  @media(max-width:520px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="card">
  <h1>Employee Attrition Predictor</h1>
  <p class="sub">Will the employee leave or stay?</p>
  <form method="post" action="/">
    <div class="grid">
      <div><label>Education</label>
        <select name="Education">{% for o in ['Bachelors','Masters','PHD'] %}<option {{ 'selected' if form.get('Education')==o }}>{{o}}</option>{% endfor %}</select></div>
      <div><label>Joining Year</label>
        <input type="number" name="JoiningYear" min="1990" max="2100" value="{{ form.get('JoiningYear','2016') }}" required></div>
      <div><label>City</label>
        <select name="City">{% for o in ['Bangalore','New Delhi','Pune'] %}<option {{ 'selected' if form.get('City')==o }}>{{o}}</option>{% endfor %}</select></div>
      <div><label>Payment Tier (1 = highest)</label>
        <select name="PaymentTier">{% for o in ['1','2','3'] %}<option {{ 'selected' if form.get('PaymentTier')==o }}>{{o}}</option>{% endfor %}</select></div>
      <div><label>Age</label>
        <input type="number" name="Age" min="18" max="70" value="{{ form.get('Age','28') }}" required></div>
      <div><label>Gender</label>
        <select name="Gender">{% for o in ['Male','Female'] %}<option {{ 'selected' if form.get('Gender')==o }}>{{o}}</option>{% endfor %}</select></div>
      <div><label>Ever Benched</label>
        <select name="EverBenched">{% for o in ['No','Yes'] %}<option {{ 'selected' if form.get('EverBenched')==o }}>{{o}}</option>{% endfor %}</select></div>
      <div><label>Experience in Current Domain (yrs)</label>
        <input type="number" name="ExperienceInCurrentDomain" min="0" max="50" value="{{ form.get('ExperienceInCurrentDomain','2') }}" required></div>
    </div>
    <button type="submit">Predict</button>
  </form>

  {% if error %}
    <div class="result error"><h2>Error</h2>{{ error }}</div>
  {% elif result %}
    <div class="result {{ 'leave' if 'Leave' in result.prediction else 'stay' }}">
      <h2>{{ result.prediction }}</h2>
      Leave: {{ result.leave_probability }}% &nbsp;|&nbsp; Stay: {{ result.stay_probability }}%
    </div>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = predict(form)
        except (ValueError, TypeError) as e:
            error = str(e)
    return render_template_string(PAGE, result=result, error=error, form=form)


# ---------------------------------------------------------------------------
# JSON API:  POST /predict   (Content-Type: application/json)
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400
    try:
        return jsonify(predict(data))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local run only. On Render, gunicorn starts the app.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
