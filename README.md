# 👥 Employee Attrition Predictor

An interactive **Machine Learning web application** that predicts whether an employee is likely to stay with or leave an organization using a trained Gaussian Naive Bayes classifier.

🚀 **Live Demo:** https://employee-churn-prediction-1.onrender.com

📂 **GitHub Repository:** https://github.com/Pranav-Sandbhor-008/employee_churn_prediction

---

## ✨ Project Overview

Employee attrition is an important challenge for organizations. Understanding potential retention risks can help HR teams make better-informed decisions about employee engagement, career development, and workforce planning.

The Employee Attrition Predictor provides an interactive dashboard where users can adjust employee attributes and receive an estimated probability of leaving, a predicted outcome, risk classification, feature-effect visualizations, and suggested HR actions.

The application combines a Flask backend, a pre-trained machine learning model, and a responsive web interface.

## 🚀 Key Features

- **Employee Attrition Prediction:** Predict whether an employee is likely to stay or leave.
- **Probability Gauge:** Visualize the model-estimated probability of leaving.
- **Risk Classification:** Categorize predictions as Low, Medium, or High risk.
- **Feature-Effect Visualization:** Explore the model's feature-level log-likelihood differences.
- **Suggested HR Actions:** Display suggestions based on the predicted risk category.
- **Interactive Employee Profile:** Adjust employee attributes using sliders and selection buttons.
- **Prediction Presets:** Quickly test predefined employee profiles.
- **Prediction History:** Save and compare predictions during the current browser session.
- **Dark and Light Themes:** Switch between display themes.
- **REST API:** Generate predictions and retrieve model metadata through Flask endpoints.
- **Render Deployment:** Host the application as a web service.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and prediction logic |
| Flask | Web framework and REST API |
| Scikit-learn | Gaussian Naive Bayes model |
| Pandas | Preparing model input data |
| NumPy | Numerical calculations |
| HTML | Dashboard structure |
| CSS | Responsive styling and themes |
| JavaScript | Interactive controls and API requests |
| Gunicorn | Production WSGI server |
| Render | Application hosting |

## 🤖 Machine Learning Model

The project uses a pre-trained **Gaussian Naive Bayes classifier** stored in a pickle file.

The model estimates the probability of employee attrition based on the eight input attributes. The application displays the predicted outcome as `Likely to Leave` or `Likely to Stay`.

### Model input features

| Feature | Description |
|---|---|
| `Education` | Education level |
| `JoiningYear` | Year the employee joined |
| `City` | Employee's associated city |
| `PaymentTier` | Payment tier, from 1 to 3 |
| `Age` | Employee age |
| `Gender` | Gender value recorded in the dataset |
| `EverBenched` | Whether the employee was ever benched |
| `ExperienceInCurrentDomain` | Years of experience in the current domain |

### Prediction and risk categories

- **Likely to Stay:** Estimated leave probability below 50%.
- **Likely to Leave:** Estimated leave probability of 50% or higher.
- **Low Risk:** Leave probability below 35%.
- **Medium Risk:** Leave probability from 35% to below 60%.
- **High Risk:** Leave probability of 60% or higher.

The prediction threshold and risk categories are application rules, not guarantees of an employee's future behavior.

## 📁 Project Structure

The repository should contain the following files:

```text
employee_churn_prediction/
│
├── app.py
├── naive.pkl
├── requirements.txt
├── README.md
│
└── .gitignore
```

The `naive.pkl` file contains the serialized trained model. Keep the model filename consistent with the path specified in `app.py`.

## ⚙️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pranav-Sandbhor-008/employee_churn_prediction.git
cd employee_churn_prediction
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS or Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the model file

Place your trained model in the project directory and name it:

```text
naive.pkl
```

Your uploaded file is currently named `naive(3).pkl`. Rename it to `naive.pkl` before running the uploaded `app(4).py` code, and save that Python file as `app.py`.

### 5. Start the application

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:5000

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```text
Flask
numpy
pandas
scikit-learn
gunicorn
```

**Compatibility note:** Use a Scikit-learn version compatible with the one used to serialize the model. If the original training environment is available, check its Scikit-learn version and reproduce that environment for deployment.

## 🌐 REST API Endpoints

The Flask application exposes the following endpoints.

### 1. Home — `/`

**Method:** `GET`

Returns the interactive employee attrition dashboard.

### 2. Predict — `/predict`

**Method:** `POST`

Accepts employee attributes as a JSON object and returns the prediction, leave and stay probabilities, risk category, feature effects, and suggested HR actions.

Example request:

```json
{
  "Education": "Bachelors",
  "JoiningYear": 2017,
  "City": "Pune",
  "PaymentTier": 3,
  "Age": 24,
  "Gender": "Female",
  "EverBenched": "Yes",
  "ExperienceInCurrentDomain": 1
}
```

Example response structure:

```json
{
  "prediction": "Likely to Leave",
  "leave_probability": 65.25,
  "stay_probability": 34.75,
  "risk": "High",
  "factors": [],
  "advice": [
    "Prioritise a retention conversation this week."
  ]
}
```

*The response above is illustrative; actual values depend on the model's prediction for the submitted employee profile.*

### 3. Model Metadata — `/meta`

**Method:** `GET`

Returns model metadata, including the sample count recorded by the model, class priors, and number of input features.

### 4. Health Check — `/health`

**Method:** `GET`

Returns a simple JSON response indicating that the application is running.

## ☁️ Deploy on Render

The project can be hosted using Render's Web Service functionality.

1. Push `app.py`, `naive.pkl`, `requirements.txt`, and the README to your GitHub repository.
2. Open [Render](https://render.com/).
3. Create a new **Web Service**.
4. Connect your `employee_churn_prediction` GitHub repository.
5. Configure the following commands:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
gunicorn app:app
```

6. Deploy the service and check the application logs if any dependency or model-loading errors occur.

**Live application:** https://employee-churn-prediction-1.onrender.com

## 🔮 Future Improvements

- Evaluate model performance using precision, recall, F1-score, and a confusion matrix.
- Add explainable AI techniques for more reliable interpretation.
- Support batch predictions using CSV uploads.
- Add model comparison and evaluation dashboards.
- Improve API documentation and automated testing.
- Add safeguards for fair and responsible use of employee-related predictions.

## 👨‍💻 Author

**Pranav Sandbhor**

- **GitHub:** https://github.com/Pranav-Sandbhor-008
- **Project Repository:** https://github.com/Pranav-Sandbhor-008/employee_churn_prediction
- **Live Demo:** https://employee-churn-prediction-1.onrender.com
- **LinkedIn :** https://www.linkedin.com/in/pranavsandbhor/

## ⭐ Support

If you find this project useful, consider starring the repository and sharing feedback.

---

*Built with Python, Flask, and Machine Learning.*
