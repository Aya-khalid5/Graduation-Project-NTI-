"""
Global Food Waste Economic Impact Predictor
Graduation Project — Streamlit Web Application

This app deploys the exact Linear Regression and Logistic Regression models
trained in FinalProject.ipynb, using the exact preprocessing pipeline
(OneHotEncoder + StandardScaler) reproduced from that notebook.

Models and preprocessing objects are NOT retrained here — they are loaded
from disk (models/*.pkl) and applied as-is.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Food Waste Economic Impact Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
  
    .main { background-color: #f7f9fb; }
    .block-container { padding-top: 2rem; }

    .app-hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 45%, #40916c 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        
    }
    .app-hero h1 { margin-bottom: 0.3rem; font-size: 2.1rem; }
    .app-hero p { font-size: 1.05rem; opacity: 0.92; margin: 0.15rem 0; }

    .card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eef1f4;
        margin-bottom: 1.1rem;
        color: #1a1a1a !important;
    }
    .card h3 {
        margin-top: 0;
        color: #1a1a1a !important;
    }

    .result-box {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-box h2 { margin: 0; font-size: 2.1rem; }
    .result-box p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    .badge-low {
        background:#d8f3dc; color:#1b4332; padding:6px 16px;
        border-radius:20px; font-weight:600; display:inline-block;
    }
    .badge-high {
        background:#ffe5d9; color:#7c2d12; padding:6px 16px;
        border-radius:20px; font-weight:600; display:inline-block;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression_model.pkl"))
    logistic_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return linear_model, logistic_model, encoder, scaler


@st.cache_data
def load_metadata():
    with open(os.path.join(ASSETS_DIR, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_asset_csv(name):
    return pd.read_csv(os.path.join(ASSETS_DIR, name))


try:
    linear_model, logistic_model, encoder, scaler = load_models()
    meta = load_metadata()
    MODELS_OK = True
    LOAD_ERROR = None
except Exception as e:  # noqa: BLE001
    MODELS_OK = False
    LOAD_ERROR = str(e)

NUM_COLS = meta["num_cols"] if MODELS_OK else []
CAT_COLS = meta["cat_cols"] if MODELS_OK else []
COUNTRIES = meta["countries"] if MODELS_OK else []
FOOD_CATEGORIES = meta["food_categories"] if MODELS_OK else []


# --------------------------------------------------------------------------
# PREPROCESSING (mirrors the notebook exactly — no refitting on user input)
# --------------------------------------------------------------------------
def build_feature_vector(country, year, food_category, total_waste,
                          avg_waste_capita, population, household_waste):
    """Builds the exact 33-feature vector used at training time:
    [scaled numeric features] + [one-hot encoded Country, Food Category]
    Order: Year, Total Waste (Tons), Avg Waste per Capita (Kg),
           Population (Million), Household Waste (%) -> then encoder output.
    """
    row = pd.DataFrame(
        [{
            "Country": country,
            "Year": year,
            "Food Category": food_category,
            "Total Waste (Tons)": total_waste,
            "Avg Waste per Capita (Kg)": avg_waste_capita,
            "Population (Million)": population,
            "Household Waste (%)": household_waste,
        }]
    )

    num_part = scaler.transform(row[NUM_COLS])
    cat_part = encoder.transform(row[CAT_COLS])
    final_vector = np.hstack([num_part, cat_part])
    return final_vector


def validate_inputs(year, total_waste, avg_waste_capita, population, household_waste):
    errors = []
    if total_waste is None or total_waste <= 0:
        errors.append("Total Waste (Tons) must be a positive number.")
    if avg_waste_capita is None or avg_waste_capita <= 0:
        errors.append("Avg Waste per Capita (Kg) must be a positive number.")
    if population is None or population <= 0:
        errors.append("Population (Million) must be a positive number.")
    if household_waste is None or not (0 <= household_waste <= 100):
        errors.append("Household Waste (%) must be between 0 and 100.")
    if year is None:
        errors.append("Year is required.")
    return errors


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.markdown("##  Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Prediction", "📊 Model Information", "📈 Visualizations"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Graduation Project**
    Global Food Waste — Economic Impact
    Machine Learning Deployment

    Built with Streamlit · scikit-learn
    """
)

if not MODELS_OK:
    st.error(
        "⚠️ Failed to load model/preprocessing files from the `models/` and "
        f"`assets/` folders. Details: {LOAD_ERROR}"
    )
    st.stop()


# --------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="app-hero">
            <h1>🌍 Global Food Waste — Economic Impact Predictor</h1>
            <p>A Machine Learning powered decision-support tool analyzing food waste
            data across 20 countries and 8 food categories.</p>
            <p>Graduation Project · Linear &amp; Logistic Regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card"><h3> Countries</h3><h2>20</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3> Food Categories</h3><h2>8</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3> Years Covered</h3><h2>{meta["year_min"]}–{meta["year_max"]}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card"><h3> ML Models</h3><h2>2</h2></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
            <h3> Problem Statement</h3>
            <p>Food waste is a growing global crisis with significant economic
            consequences. Governments, retailers, and organizations need a way
            to <b>anticipate the economic loss</b> caused by food waste and to
            <b>classify the severity</b> of that loss so that resources and
            interventions can be prioritized effectively.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
            <h3> Project Objective</h3>
            <ul>
                <li>Predict the <b>Economic Loss (Million $)</b> caused by food
                waste using a <b>Linear Regression</b> model.</li>
                <li>Classify records into <b>Low Loss</b> / <b>High Loss</b>
                categories (relative to the training median) using a
                <b>Logistic Regression</b> model.</li>
                <li>Deploy both models in an interactive, presentation-ready
                web application.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
            <h3> Machine Learning Approach</h3>
            <ol>
                <li><b>Data Cleaning:</b> group-wise median imputation for
                numeric columns (Country+Year, Country+Food Category,
                Country), mode imputation for the categorical column.</li>
                <li><b>Train/Test Split:</b> 80/20 split, <code>random_state=42</code>,
                performed <i>before</i> imputation to avoid data leakage.</li>
                <li><b>Encoding:</b> <code>OneHotEncoder(handle_unknown="ignore")</code>
                on <i>Country</i> and <i>Food Category</i>.</li>
                <li><b>Scaling:</b> <code>StandardScaler</code> on the five
                numeric features.</li>
                <li><b>Modeling:</b> <code>LinearRegression</code> for the
                continuous target and <code>LogisticRegression</code> for the
                binarized target (High Loss = Economic Loss ≥ training
                median).</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
            <h3> Input Features</h3>
            <p>Country · Year · Food Category · Total Waste (Tons) ·
            Avg Waste per Capita (Kg) · Population (Million) ·
            Household Waste (%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("Use the sidebar to navigate to **Prediction**, **Model Information**, or **Visualizations**.")


# --------------------------------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------------------------------
elif page == "🔮 Prediction":
    st.markdown("##  Make a Prediction")
    st.caption("All inputs are transformed using the exact fitted encoder and scaler from training — no refitting occurs here.")

    with st.form("prediction_form"):
        st.markdown("#### Input Features")
        f1, f2, f3 = st.columns(3)
        with f1:
            country = st.selectbox("Country", COUNTRIES)
            year = st.number_input(
                "Year", min_value=meta["year_min"], max_value=meta["year_max"] + 5,
                value=meta["year_max"], step=1,
            )
        with f2:
            food_category = st.selectbox("Food Category", FOOD_CATEGORIES)
            total_waste = st.number_input(
                "Total Waste (Tons)", min_value=0.0,
                value=float(round(np.mean(meta["ranges"]["Total Waste (Tons)"]), 2)),
                step=100.0, format="%.2f",
            )
            with f3:
            avg_waste_capita = st.number_input(
                "Avg Waste per Capita (Kg)", min_value=0.0,
                value=float(round(np.mean(meta["ranges"]["Avg Waste per Capita (Kg)"]), 2)),
                step=1.0, format="%.2f",
            )
            population = st.number_input(
                "Population (Million)", min_value=0.0,
                value=float(round(np.mean(meta["ranges"]["Population (Million)"]), 2)),
                step=1.0, format="%.2f",
            )
            with st.container():
            st.markdown("""
                <style>
                div[data-testid="stSlider"], div[data-testid="stSlider"] * {
                    direction: ltr !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            household_waste = st.slider(
                "Household Waste (%)", min_value=0.0, max_value=100.0,
                value=50.0,
                step=0.1,
            )

        st.markdown("#### Model Selection")
        model_choice = st.radio(
            "Choose a model",
            ["Linear Regression (predict Economic Loss)", "Logistic Regression (classify Loss Level)"],
            horizontal=True,
        )

        submitted = st.form_submit_button(" Predict", use_container_width=True)

    if submitted:
        errors = validate_inputs(year, total_waste, avg_waste_capita, population, household_waste)

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                features = build_feature_vector(
                    country, year, food_category, total_waste,
                    avg_waste_capita, population, household_waste,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Preprocessing failed: {e}")
                st.stop()

            if "Linear" in model_choice:
                pred = linear_model.predict(features)[0]
                st.markdown(
                    f"""
                    <div class="result-box">
                        <p>Predicted Economic Loss</p>
                        <h2>${pred:,.2f} Million</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                pred_class = int(logistic_model.predict(features)[0])
                label = "High Loss" if pred_class == 1 else "Low Loss"
                badge_class = "badge-high" if pred_class == 1 else "badge-low"

                proba_html = ""
                if hasattr(logistic_model, "predict_proba"):
                    proba = logistic_model.predict_proba(features)[0]
                    proba_html = f"<p>Confidence: Low Loss {proba[0]*100:.1f}% · High Loss {proba[1]*100:.1f}%</p>"

                st.markdown(
                    f"""
                    <div class="result-box">
                        <p>Predicted Class</p>
                        <h2><span class="{badge_class}">{label}</span></h2>
                        {proba_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if hasattr(logistic_model, "predict_proba"):
                    proba = logistic_model.predict_proba(features)[0]
                    fig = go.Figure(go.Bar(
                        x=["Low Loss", "High Loss"], y=[proba[0], proba[1]],
                        marker_color=["#40916c", "#e76f51"], text=[f"{p*100:.1f}%" for p in proba],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Class Probability", yaxis_title="Probability",
                        yaxis_range=[0, 1], height=350, margin=dict(t=50, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Reference: training-set median Economic Loss = "
                f"${meta['median_loss']:,.2f} Million (used as the High/Low Loss threshold)."
            )


# --------------------------------------------------------------------------
# MODEL INFORMATION PAGE
# --------------------------------------------------------------------------
elif page == "📊 Model Information":
    st.markdown("##  Model Information & Evaluation Metrics")

    tab1, tab2 = st.tabs([" Linear Regression", " Logistic Regression"])

    with tab1:
        st.markdown("#### Predicts: Economic Loss (Million $)")
        lm = meta["linear_metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{lm['MAE']:,.2f}")
        m2.metric("MSE", f"{lm['MSE']:,.2f}")
        m3.metric("RMSE", f"{lm['RMSE']:,.2f}")
        m4.metric("R² Score", f"{lm['R2']:.4f}")
        st.markdown(
            """
            <div class="card">
            <p><b>Model:</b> <code>sklearn.linear_model.LinearRegression</code></p>
            <p><b>Features:</b> 33 (5 scaled numeric + 20 one-hot Country + 8 one-hot Food Category)</p>
            <p>R² indicates the proportion of variance in Economic Loss explained by the model on the held-out test set.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("#### Classifies: Low Loss vs High Loss (threshold = training median)")
        gm = meta["logistic_metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{gm['Accuracy']*100:.2f}%")
        m2.metric("Precision", f"{gm['Precision']:.4f}")
        m3.metric("Recall", f"{gm['Recall']:.4f}")
        m4.metric("F1 Score", f"{gm['F1']:.4f}")
        st.markdown(
            f"""
            <div class="card">
            <p><b>Model:</b> <code>sklearn.linear_model.LogisticRegression(max_iter=1000)</code></p>
            <p><b>Target definition:</b> High Loss = 1 if Economic Loss ≥ training median
            (${meta['median_loss']:,.2f}M), else Low Loss = 0.</p>
            <p><b>Features:</b> same 33-feature vector as the Linear Regression model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cm = np.array(gm["confusion_matrix"])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Greens",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=gm["labels"], y=gm["labels"],
        )
        fig_cm.update_layout(title="Confusion Matrix (Test Set)", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)


# --------------------------------------------------------------------------
# VISUALIZATIONS PAGE
# --------------------------------------------------------------------------
elif page == "📈 Visualizations":
    st.markdown("##  Visualizations & Business Insights")

    v1, v2 = st.tabs([" Model Performance", " Business Insights"])

    with v1:
        st.markdown("#### Model Performance Comparison")
        lm = meta["linear_metrics"]
        gm = meta["logistic_metrics"]
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            fig = go.Figure(go.Bar(
                x=["R² Score"], y=[lm["R2"]], marker_color="#2d6a4f",
                text=[f"{lm['R2']:.3f}"], textposition="outside",
            ))
            fig.update_layout(title="Linear Regression — R² Score", yaxis_range=[0, 1], height=350)
            st.plotly_chart(fig, use_container_width=True)

        with comp_col2:
            fig = go.Figure(go.Bar(
                x=["Accuracy", "Precision", "Recall", "F1"],
                y=[gm["Accuracy"], gm["Precision"], gm["Recall"], gm["F1"]],
                marker_color="#40916c",
                text=[f"{v:.3f}" for v in [gm["Accuracy"], gm["Precision"], gm["Recall"], gm["F1"]]],
                textposition="outside",
            ))
            fig.update_layout(title="Logistic Regression — Classification Metrics", yaxis_range=[0, 1], height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Actual vs Predicted — Economic Loss (Test Set)")
        avp = load_asset_csv("actual_vs_predicted.csv")
        fig_avp = px.scatter(
            avp, x="actual", y="predicted", opacity=0.5,
            labels={"actual": "Actual Economic Loss (Million $)", "predicted": "Predicted Economic Loss (Million $)"},
        )
        max_val = float(max(avp["actual"].max(), avp["predicted"].max()))
        fig_avp.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color="#e76f51", dash="dash"), name="Perfect Prediction",
        ))
        fig_avp.update_layout(height=450)
        st.plotly_chart(fig_avp, use_container_width=True)

        st.markdown("#### Top Feature Influences — Linear Regression Coefficients")
        coefs = load_asset_csv("linear_coefficients.csv")
        coefs["abs_coef"] = coefs["coefficient"].abs()
        top_coefs = coefs.sort_values("abs_coef", ascending=False).head(12).sort_values("coefficient")
        fig_coef = px.bar(
            top_coefs, x="coefficient", y="feature", orientation="h",
            color="coefficient", color_continuous_scale="RdYlGn",
            labels={"coefficient": "Coefficient", "feature": "Feature"},
        )
        fig_coef.update_layout(title="Top 12 Most Influential Features", height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)

    with v2:
        st.markdown("#### Total Economic Loss by Country")
        by_country = load_asset_csv("loss_by_country.csv")
        fig_c = px.bar(
            by_country, x="Country", y="Economic Loss (Million $)",
            color="Economic Loss (Million $)", color_continuous_scale="Greens",
        )
        fig_c.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Total Economic Loss by Food Category")
            by_cat = load_asset_csv("loss_by_category.csv")
            fig_cat = px.pie(by_cat, names="Food Category", values="Economic Loss (Million $)", hole=0.45)
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_b:
            st.markdown("#### Total Economic Loss Trend by Year")
            by_year = load_asset_csv("loss_by_year.csv")
            fig_y = px.line(by_year, x="Year", y="Economic Loss (Million $)", markers=True)
            fig_y.update_layout(height=400)
            st.plotly_chart(fig_y, use_container_width=True)

        st.caption("Aggregations computed from the original (uncleaned) dataset — for exploratory/business context only; not used by the models.")
