#  Global Food Waste — Economic Impact Predictor

A deployment-ready Streamlit web application for a Graduation Project that predicts
the **economic loss** caused by food waste and **classifies** its severity, using the
exact models and preprocessing pipeline developed in `FinalProject.ipynb`.

> **Source of truth:** `FinalProject.ipynb`. This app does **not** retrain or modify
> the models — it loads the already-trained `.pkl` files and reproduces the exact
> preprocessing (OneHotEncoder + StandardScaler) fitted in the notebook.

---

##  Project Structure

```
GraduationProject/
│
├── FinalProject.ipynb              # Original notebook (source of truth)
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # One-command container startup
├── .dockerignore                   # Files excluded from the image
├── README.md                       # This file
│
├── models/
│   ├── linear_regression_model.pkl # Trained LinearRegression (from notebook)
│   ├── logistic_regression_model.pkl # Trained LogisticRegression (from notebook)
│   ├── encoder.pkl                 # Fitted OneHotEncoder (Country, Food Category)
│   └── scaler.pkl                  # Fitted StandardScaler (5 numeric features)
│
└── assets/
    ├── metadata.json               # Dropdown values, feature ranges, eval metrics
    ├── linear_coefficients.csv     # Linear model coefficients (feature importance)
    ├── actual_vs_predicted.csv     # Test-set actual vs. predicted values
    ├── loss_by_country.csv         # Business-insight aggregation
    ├── loss_by_category.csv        # Business-insight aggregation
    └── loss_by_year.csv            # Business-insight aggregation
```

---

##  Important note on `encoder.pkl` / `scaler.pkl`

`FinalProject.ipynb` fits an `OneHotEncoder` and a `StandardScaler` on `X_train`, but
only saves the two trained **models** with `joblib.dump` — it never persists the
encoder/scaler objects themselves. Since the notebook's train/test split uses a fixed
`random_state=42` and a fully deterministic preprocessing sequence, that exact same
sequence was re-run **once**, from the notebook's own code, to regenerate an
identical fitted `encoder.pkl` and `scaler.pkl`. This is not "refitting on user
input" — it is reconstructing the two objects the notebook computed but didn't save,
using the same data and same code. This was verified by confirming the reconstructed
encoder/scaler reproduce sensible, consistent metrics when combined with the
already-trained models on the test set (see the metrics reported in the app's
**Model Information** page). At prediction time, the app only ever calls
`encoder.transform(...)` / `scaler.transform(...)` — never `.fit()` — on user input.

---

##  Modeling Summary (from the notebook)

| Item | Detail |
|---|---|
| Input features | Country, Year, Food Category, Total Waste (Tons), Avg Waste per Capita (Kg), Population (Million), Household Waste (%) |
| Regression target | `Economic Loss (Million $)` |
| Classification target | `1` (High Loss) if Economic Loss ≥ **training-set median**, else `0` (Low Loss) |
| Split | 80/20, `train_test_split(..., random_state=42)`, done **before** imputation |
| Missing-value handling | Group-median imputation (train-set statistics only) + mode for the categorical column |
| Encoding | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on Country + Food Category |
| Scaling | `StandardScaler()` on the 5 numeric features |
| Feature order | `[Year, Total Waste, Avg Waste/Capita, Population, Household Waste]` + one-hot(Country) + one-hot(Food Category) = **33 features** |
| Models | `LinearRegression()`, `LogisticRegression(max_iter=1000)` |

---

##  Application Pages

1. **Home** — project overview, problem statement, objective, ML approach.
2. **Prediction** — enter feature values, choose a model, get a live prediction
   (with class probabilities for Logistic Regression).
3. **Model Information** — MAE / MSE / RMSE / R² (Linear) and Accuracy / Precision /
   Recall / F1 + confusion matrix (Logistic).
4. **Visualizations** — model performance charts, actual-vs-predicted scatter plot,
   top feature coefficients, and business-insight aggregations (loss by country,
   category, and year).

---

##  Testing Performed

Before delivery, the following were verified programmatically:

- Loading of all four `.pkl` files (models, encoder, scaler) 
- The reconstructed encoder/scaler + saved models reproduce coherent evaluation
  metrics on the notebook's test split (R² ≈ 0.53, Accuracy ≈ 0.78) 
- End-to-end feature-vector construction for both models across **all 160**
  Country × Food Category combinations — no errors 
- Unseen/unknown categorical values are handled gracefully via
  `handle_unknown="ignore"` (encoded as all-zero, no crash) 
- Boundary and extreme numeric inputs (e.g. future years, max population) 
- Input validation rejects non-positive waste/population values and out-of-range
  percentages 
- `app.py` parses with no syntax errors 

> Note: this sandbox has no internet access, so `streamlit run app.py` itself could
> not be executed here. All underlying prediction/preprocessing logic used by the
> app was tested directly and works correctly; please do a final
> `streamlit run app.py` smoke test locally before presenting.

---

##  Run Locally

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

##  Run with Docker

The project includes a `Dockerfile`, `.dockerignore`, and `docker-compose.yml`.

### Option A — Docker Compose (recommended)

```bash
cd GraduationProject
docker compose up --build
```

Open **http://localhost:8501**. Stop with `Ctrl+C`, or `docker compose down` to remove the container.

### Option B — Plain Docker

```bash
cd GraduationProject

# Build the image
docker build -t food-waste-app .

# Run the container
docker run -p 8501:8501 food-waste-app
```

Open **http://localhost:8501**.

> The image installs dependencies from `requirements.txt` (pinned to
> `scikit-learn==1.9.0` to match the version the models were trained with) and
> copies in `app.py`, `models/`, and `assets/`. No retraining happens inside the
> container — it only loads and serves the existing `.pkl` files.

---

##  GitHub Setup

```bash
cd GraduationProject
git init
git add .
git commit -m "Graduation Project: Food Waste Economic Impact Predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Tip: the `models/` and `assets/` folders are small (few KB each) — no need for
> Git LFS.

---

##  Deploy to Streamlit Community Cloud

1. Push this project to a public (or Streamlit-authorized private) GitHub repo, as above.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch
   the app automatically.
6. Once live, share the generated `https://<your-app>.streamlit.app` URL.

---

##  Tech Stack

- **Streamlit** — web application framework
- **scikit-learn** — trained models & preprocessing (loaded, not retrained)
- **Plotly** — interactive charts
- **Pandas / NumPy** — data handling
- **Joblib** — model/preprocessing object persistence
- 
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/40bd3425-d63a-4df8-b677-ed1e46f6c627" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/4b14256d-92ed-4430-8f43-3d387b9b3d00" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/b28d87b8-c8c5-4cb9-be8e-f7047294375d" />
Presention : https://canva.link/4wb5g0si2z6wqtu
Docker Hub: https://hub.docker.com/layers/ayakhaled5/food-waste-app/latest/images/sha256%3A43573111205e40a236f215be909f35578f951a4c9a78ae8fb214517d24ee2b9c?uuid=2609A4C3-D5B4-46D9-8D45-EEACB81E7277

