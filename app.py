"""
Employee Attrition Predictor - Flask app (Render-ready)
Model: Gaussian Naive Bayes (naive.pkl)  ->  Likely to Leave / Likely to Stay
"""
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "naive.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["Education", "JoiningYear", "City", "PaymentTier",
            "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain"]
LABELS = {
    "Education": "Education", "JoiningYear": "Joining year", "City": "City",
    "PaymentTier": "Payment tier", "Age": "Age", "Gender": "Gender",
    "EverBenched": "Ever benched", "ExperienceInCurrentDomain": "Domain experience",
}
# Alphabetical label-encoding used in training
EDUCATION = {"Bachelors": 0, "Masters": 1, "PHD": 2}
CITY = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}
GENDER = {"Female": 0, "Male": 1}
BENCHED = {"No": 0, "Yes": 1}

ADVICE = {
    "Low": ["Keep regular check-ins and recognise good work.",
            "Share a clear growth path to sustain engagement."],
    "Medium": ["Hold a 1:1 to understand motivation and concerns.",
               "Review pay tier, workload and upcoming career steps.",
               "Offer learning or mentoring opportunities."],
    "High": ["Prioritise a retention conversation this week.",
             "Review compensation tier and promotion timeline.",
             "Check project allocation and bench time.",
             "Assign a mentor and define a visible growth plan."],
}


def _lookup(mapping, value, field):
    if isinstance(value, str):
        for key, code in mapping.items():
            if key.lower() == value.strip().lower():
                return code
        if value.strip().isdigit() and int(value) in mapping.values():
            return int(value)
    elif isinstance(value, (int, float)) and int(value) in mapping.values():
        return int(value)
    raise ValueError(f"Invalid {field}: {value!r}. Allowed: {list(mapping)}")


def build_row(data):
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
    if not 0 <= row["ExperienceInCurrentDomain"] <= 50:
        raise ValueError("ExperienceInCurrentDomain must be between 0 and 50.")
    return pd.DataFrame([row], columns=FEATURES)


def factor_effects(X):
    """Per-feature log-likelihood ratio (Leave vs Stay). Positive = pushes to Leave."""
    x = X.values[0].astype(float)
    th, var = model.theta_, model.var_
    ll = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - th) ** 2) / var
    return (ll[1] - ll[0]).tolist()


def predict(data):
    X = build_row(data)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    leave = float(proba[classes.index(1)]) * 100
    risk = "Low" if leave < 35 else "Medium" if leave < 60 else "High"
    effects = factor_effects(X)
    factors = [{"feature": LABELS[f], "value": str(data[f]), "effect": round(e, 3)}
               for f, e in zip(FEATURES, effects)]
    factors.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return {
        "prediction": "Likely to Leave" if leave >= 50 else "Likely to Stay",
        "leave_probability": round(leave, 2),
        "stay_probability": round(100 - leave, 2),
        "risk": risk,
        "factors": factors,
        "advice": ADVICE[risk],
    }


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400
    try:
        return jsonify(predict(data))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/meta")
def meta():
    pri = model.class_prior_.tolist()
    return jsonify({
        "samples": int(model.class_count_.sum()),
        "stay_prior": round(pri[0] * 100, 1),
        "leave_prior": round(pri[1] * 100, 1),
        "features": len(FEATURES),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Employee Attrition Predictor</title>
<style>
:root{--bg:#f4f5fb;--card:#fff;--text:#18181b;--mute:#6b7280;--line:#e5e7eb;--accent:#6366f1;--accent2:#8b5cf6;--good:#10b981;--warn:#f59e0b;--bad:#ef4444;--chip:#eef0ff;--shadow:0 8px 30px rgba(30,30,80,.08)}
:root[data-theme=dark]{--bg:#0e1020;--card:#171a2e;--text:#eef0ff;--mute:#98a0c0;--line:#2a2f4d;--chip:#222749;--shadow:0 8px 30px rgba(0,0,0,.4)}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);transition:background .3s}
.hero{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;padding:36px 20px 80px}
.wrap{max-width:1120px;margin:0 auto}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px}
.hero h1{margin:0;font-size:clamp(1.5rem,3.5vw,2.2rem)}
.hero p{margin:6px 0 0;opacity:.9}
.theme{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);color:#fff;border-radius:999px;padding:9px 16px;cursor:pointer;font-size:.9rem;font-weight:600;white-space:nowrap}
.theme:hover{background:rgba(255,255,255,.3)}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}
.stat{background:rgba(255,255,255,.16);border-radius:12px;padding:10px 16px;min-width:130px}
.stat b{display:block;font-size:1.3rem}.stat span{font-size:.78rem;opacity:.85}
main{margin-top:-52px;padding:0 20px 40px}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:18px;padding:22px;box-shadow:var(--shadow);margin-bottom:20px}
.card h3{margin:0 0 16px;font-size:1.05rem;display:flex;align-items:center;gap:8px}
.sec{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);margin:18px 0 10px;font-weight:700}
.sec:first-of-type{margin-top:0}
.field{margin-bottom:16px}
.field label{display:flex;justify-content:space-between;font-size:.85rem;font-weight:600;margin-bottom:7px}
.val{background:var(--chip);color:var(--accent);padding:2px 10px;border-radius:999px;font-size:.8rem}
.seg{display:flex;gap:6px;flex-wrap:wrap}
.seg button{flex:1;min-width:70px;padding:9px 10px;border:1px solid var(--line);background:transparent;color:var(--text);border-radius:10px;cursor:pointer;font-size:.88rem;transition:.15s}
.seg button:hover{border-color:var(--accent)}
.seg button.on{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border-color:transparent;box-shadow:0 4px 12px rgba(99,102,241,.35)}
input[type=range]{width:100%;accent-color:var(--accent)}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.presets button,.btn{border:1px solid var(--line);background:var(--chip);color:var(--accent);padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:600;font-size:.85rem}
.btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border:0;width:100%;padding:12px;font-size:.95rem;margin-top:6px}
.gauge{position:relative;max-width:300px;margin:0 auto}
.gauge svg{width:100%;display:block}
.gauge .track{stroke:var(--line)}
.gauge .fill{transition:stroke-dasharray .8s cubic-bezier(.2,.8,.2,1),stroke .4s}
.gtext{position:absolute;left:0;right:0;bottom:6px;text-align:center}
.gtext b{font-size:2.3rem;display:block;line-height:1}
.gtext span{font-size:.78rem;color:var(--mute)}
.verdict{text-align:center;margin-top:10px}
.verdict h2{margin:0;font-size:1.35rem}
.pill{display:inline-block;margin-top:8px;padding:4px 14px;border-radius:999px;font-size:.8rem;font-weight:700;color:#fff}
.Low{background:var(--good)}.Medium{background:var(--warn)}.High{background:var(--bad)}
.bar2{display:flex;height:10px;border-radius:99px;overflow:hidden;margin:16px 0 6px;background:var(--line)}
.bar2 i{display:block;transition:width .6s}
.legend{display:flex;justify-content:space-between;font-size:.78rem;color:var(--mute)}
.frow{display:grid;grid-template-columns:130px 1fr 52px;gap:8px;align-items:center;font-size:.82rem;margin-bottom:9px}
.frow small{color:var(--mute);display:block;font-size:.72rem}
.track2{height:8px;background:var(--line);border-radius:9px;position:relative}
.track2::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--mute);opacity:.5}
.track2 i{position:absolute;top:0;bottom:0;border-radius:9px;transition:all .5s}
.fl{background:var(--bad)}.fs{background:var(--good)}
.axis{display:flex;justify-content:space-between;font-size:.7rem;color:var(--mute);margin:-2px 0 0 138px;padding-right:60px}
ul.tips{margin:0;padding-left:18px;font-size:.88rem;line-height:1.7}
.hist{width:100%;border-collapse:collapse;font-size:.82rem}
.hist th,.hist td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}
.hist th{color:var(--mute);font-weight:600}
.empty{color:var(--mute);font-size:.85rem;text-align:center;padding:10px}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tabs button{border:0;background:transparent;color:var(--mute);padding:8px 14px;border-radius:999px;cursor:pointer;font-weight:600}
.tabs button.on{background:var(--chip);color:var(--accent)}
.tab{display:none;font-size:.9rem;line-height:1.65}.tab.on{display:block}
.fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.fg div{background:var(--chip);border-radius:12px;padding:12px 14px}
.fg b{display:block;margin-bottom:3px}.fg span{color:var(--mute);font-size:.82rem}
:root[data-theme=dark] .err{background:#3b2f10;color:#fcd34d}
.err{background:#fef3c7;color:#92400e;padding:10px 14px;border-radius:10px;font-size:.85rem;display:none;margin-top:10px}
footer{text-align:center;color:var(--mute);font-size:.78rem;padding:10px 20px 30px}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <div class="top">
    <div><h1>👥 Employee Attrition Predictor</h1><p>Adjust the profile and see the leave / stay prediction update live.</p></div>
    <button class="theme" id="themeBtn">🌙 Dark mode</button>
  </div>
  <div class="stats">
    <div class="stat"><b>GaussianNB</b><span>Model</span></div>
    <div class="stat"><b id="sSamples">–</b><span>Training records</span></div>
    <div class="stat"><b id="sFeat">8</b><span>Input features</span></div>
    <div class="stat"><b id="sLeave">–</b><span>Base leave rate</span></div>
  </div>
</div></header>

<main><div class="wrap">
<div class="grid">
  <section>
    <div class="card">
      <h3>🧾 Employee Profile</h3>
      <div class="presets">
        <button data-preset="a">⚡ Early-career, Tier 3</button>
        <button data-preset="b">🏆 Senior, Tier 1</button>
        <button data-preset="c">🔄 Reset</button>
      </div>

      <div class="sec">Personal</div>
      <div class="field"><label>Age <span class="val" id="v_Age"></span></label>
        <input type="range" id="Age" min="18" max="60"></div>
      <div class="field"><label>Gender</label>
        <div class="seg" data-field="Gender"><button data-v="Male">♂ Male</button><button data-v="Female">♀ Female</button></div></div>
      <div class="field"><label>Education</label>
        <div class="seg" data-field="Education"><button data-v="Bachelors">🎓 Bachelors</button><button data-v="Masters">📘 Masters</button><button data-v="PHD">🔬 PHD</button></div></div>

      <div class="sec">Job</div>
      <div class="field"><label>City</label>
        <div class="seg" data-field="City"><button data-v="Bangalore">Bangalore</button><button data-v="New Delhi">New Delhi</button><button data-v="Pune">Pune</button></div></div>
      <div class="field"><label>Payment tier <span class="val" style="font-weight:500">1 = highest</span></label>
        <div class="seg" data-field="PaymentTier"><button data-v="1">Tier 1</button><button data-v="2">Tier 2</button><button data-v="3">Tier 3</button></div></div>
      <div class="field"><label>Joining year <span class="val" id="v_JoiningYear"></span></label>
        <input type="range" id="JoiningYear" min="2000" max="2026"></div>
      <div class="field"><label>Experience in current domain <span class="val" id="v_ExperienceInCurrentDomain"></span></label>
        <input type="range" id="ExperienceInCurrentDomain" min="0" max="20"></div>
      <div class="field"><label>Ever benched (without project)?</label>
        <div class="seg" data-field="EverBenched"><button data-v="No">No</button><button data-v="Yes">Yes</button></div></div>

      <button class="btn primary" id="saveBtn">➕ Save this prediction to history</button>
      <div class="err" id="err"></div>
    </div>

    <div class="card">
      <h3>🕘 Prediction History <button class="btn" id="clearBtn" style="margin-left:auto">Clear</button></h3>
      <div id="hist"><div class="empty">No saved predictions yet. Save a few to compare profiles.</div></div>
    </div>
  </section>

  <section>
    <div class="card">
      <h3>📊 Prediction</h3>
      <div class="gauge">
        <svg viewBox="0 0 200 115">
          <path class="track" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke-width="16" stroke-linecap="round" pathLength="100"/>
          <path class="fill" id="arc" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="var(--good)" stroke-width="16" stroke-linecap="round" pathLength="100" stroke-dasharray="0 100"/>
        </svg>
        <div class="gtext"><b id="pct">–</b><span>probability of leaving</span></div>
      </div>
      <div class="verdict"><h2 id="verdict">Calculating…</h2><span class="pill Low" id="risk">–</span></div>
      <div class="bar2"><i id="bl" style="background:var(--bad);width:50%"></i><i id="bs" style="background:var(--good);width:50%"></i></div>
      <div class="legend"><span>🚪 Leave <b id="lp">–</b></span><span>🏠 Stay <b id="sp">–</b></span></div>
    </div>

    <div class="card">
      <h3>🔍 What's driving this result?</h3>
      <div id="factors"></div>
      <div class="axis"><span>← Stay</span><span>Leave →</span></div>
    </div>

    <div class="card">
      <h3>💡 Suggested HR actions</h3>
      <ul class="tips" id="tips"></ul>
    </div>
  </section>
</div>

<div class="card">
  <div class="tabs">
    <button class="on" data-tab="t1">About the model</button>
    <button data-tab="t2">Feature guide</button>
    <button data-tab="t3">How to read results</button>
  </div>
  <div class="tab on" id="t1">
    This tool uses a <b>Gaussian Naive Bayes</b> classifier. It compares how typical the employee's values are for people who
    <i>left</i> versus people who <i>stayed</i>, combines the evidence from all 8 features with the base leave rate, and outputs a probability.
    It is fast and transparent, but it assumes features are independent, so treat it as decision support, not a final verdict.
  </div>
  <div class="tab" id="t2"><div class="fg">
    <div><b>🎓 Education</b><span>Bachelors, Masters or PHD.</span></div>
    <div><b>📅 Joining year</b><span>Year the employee joined the company.</span></div>
    <div><b>📍 City</b><span>Bangalore, New Delhi or Pune.</span></div>
    <div><b>💰 Payment tier</b><span>1 = highest, 3 = lowest pay bracket.</span></div>
    <div><b>🎂 Age</b><span>Employee age in years.</span></div>
    <div><b>⚧ Gender</b><span>Male or Female, as recorded in the dataset.</span></div>
    <div><b>🪑 Ever benched</b><span>Kept out of projects for a month or more.</span></div>
    <div><b>🧠 Domain experience</b><span>Years of experience in the current field.</span></div>
  </div></div>
  <div class="tab" id="t3">
    <b>Risk levels:</b> Low (&lt; 35% leave), Medium (35–60%), High (&gt; 60%).<br>
    <b>Driver bars:</b> red bars push the prediction toward <i>Leave</i>, green bars toward <i>Stay</i>. Longer bars mean stronger influence for this specific profile.<br>
    Drivers show statistical association in historical data, not cause and effect. Use gender and similar attributes with care and never as the sole basis for HR decisions.
  </div>
</div>
</div></main>
<footer>Built with Flask + scikit-learn · API: POST /predict</footer>

<script>
const DEFAULT={Education:"Bachelors",JoiningYear:2016,City:"Pune",PaymentTier:"3",Age:28,Gender:"Male",EverBenched:"No",ExperienceInCurrentDomain:2};
const PRESETS={
  a:{Education:"Bachelors",JoiningYear:2017,City:"Pune",PaymentTier:"3",Age:24,Gender:"Female",EverBenched:"Yes",ExperienceInCurrentDomain:1},
  b:{Education:"Masters",JoiningYear:2012,City:"Bangalore",PaymentTier:"1",Age:38,Gender:"Male",EverBenched:"No",ExperienceInCurrentDomain:5}
};
let S={...DEFAULT}, last=null, hist=[], timer=null;
const $=id=>document.getElementById(id);
const sliders=["Age","JoiningYear","ExperienceInCurrentDomain"];

function sync(){
  document.querySelectorAll(".seg").forEach(g=>g.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.v===String(S[g.dataset.field]))));
  sliders.forEach(k=>{$(k).value=S[k];$("v_"+k).textContent=S[k]+(k=="Age"||k=="ExperienceInCurrentDomain"?" yrs":"");});
}
function schedule(){clearTimeout(timer);timer=setTimeout(run,200);}
async function run(){
  try{
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(S)});
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||"Request failed");
    $("err").style.display="none"; last=d; render(d);
  }catch(e){$("err").textContent=e.message;$("err").style.display="block";}
}
function render(d){
  const p=d.leave_probability, col=d.risk=="Low"?"var(--good)":d.risk=="Medium"?"var(--warn)":"var(--bad)";
  $("arc").setAttribute("stroke-dasharray",p+" 100"); $("arc").setAttribute("stroke",col);
  $("pct").textContent=p.toFixed(1)+"%"; $("pct").style.color=col;
  $("verdict").textContent=(d.prediction=="Likely to Leave"?"🚪 ":"🏠 ")+d.prediction;
  $("risk").textContent=d.risk+" risk"; $("risk").className="pill "+d.risk;
  $("bl").style.width=p+"%"; $("bs").style.width=(100-p)+"%";
  $("lp").textContent=p.toFixed(1)+"%"; $("sp").textContent=d.stay_probability.toFixed(1)+"%";
  const max=Math.max(...d.factors.map(f=>Math.abs(f.effect)),0.001);
  $("factors").innerHTML=d.factors.slice(0,6).map(f=>{
    const w=Math.abs(f.effect)/max*50, left=f.effect>0;
    return `<div class="frow"><div>${f.feature}<small>${f.value}</small></div>
      <div class="track2"><i class="${left?"fl":"fs"}" style="${left?"left:50%":"right:50%"};width:${w}%"></i></div>
      <div style="text-align:right;color:${left?"var(--bad)":"var(--good)"}">${f.effect>0?"+":""}${f.effect.toFixed(1)}</div></div>`}).join("");
  $("tips").innerHTML=d.advice.map(t=>`<li>${t}</li>`).join("");
}
function renderHist(){
  if(!hist.length){$("hist").innerHTML='<div class="empty">No saved predictions yet. Save a few to compare profiles.</div>';return;}
  $("hist").innerHTML='<table class="hist"><tr><th>#</th><th>Profile</th><th>Leave %</th><th>Result</th></tr>'+
    hist.map((h,i)=>`<tr><td>${i+1}</td><td>${h.s.Age}y · ${h.s.City} · T${h.s.PaymentTier} · ${h.s.Education}</td><td>${h.p.toFixed(1)}%</td><td><span class="pill ${h.risk}">${h.risk}</span></td></tr>`).join("")+'</table>';
}

document.querySelectorAll(".seg").forEach(g=>g.addEventListener("click",e=>{
  const b=e.target.closest("button"); if(!b) return; S[g.dataset.field]=b.dataset.v; sync(); schedule();}));
sliders.forEach(k=>$(k).addEventListener("input",e=>{S[k]=+e.target.value;sync();schedule();}));
document.querySelectorAll("[data-preset]").forEach(b=>b.addEventListener("click",()=>{
  S={...(b.dataset.preset=="c"?DEFAULT:PRESETS[b.dataset.preset])};sync();run();}));
$("saveBtn").addEventListener("click",()=>{if(!last)return;hist.unshift({s:{...S},p:last.leave_probability,risk:last.risk});hist=hist.slice(0,8);renderHist();});
$("clearBtn").addEventListener("click",()=>{hist=[];renderHist();});
document.querySelectorAll(".tabs button").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll(".tabs button,.tab").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");$(b.dataset.tab).classList.add("on");}));
function setTheme(t,save){
  document.documentElement.dataset.theme=t;
  $("themeBtn").textContent=t=="dark"?"☀️ Light mode":"🌙 Dark mode";
  if(save){try{localStorage.setItem("theme",t);}catch(e){}}
}
$("themeBtn").addEventListener("click",()=>
  setTheme(document.documentElement.dataset.theme=="dark"?"light":"dark",true));
(function(){
  let t=null; try{t=localStorage.getItem("theme");}catch(e){}
  setTheme(t||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"),false);
})();

fetch("/meta").then(r=>r.json()).then(m=>{
  $("sSamples").textContent=m.samples.toLocaleString();$("sFeat").textContent=m.features;$("sLeave").textContent=m.leave_prior+"%";});
sync(); run();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
