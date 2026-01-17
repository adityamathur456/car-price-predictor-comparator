# Streamlit Advanced Car Price Prediction & Comparison App
# -------------------------------------------------
# Run: streamlit run streamlit_car_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =======================
# Config
# =======================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "catboost_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "Cleaned_Car_data.csv"


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# =======================
# Proper Theme Switching (Streamlit native) – toggle based
# =======================

def apply_theme(theme):
    if theme == "Dark":
        st._config.set_option("theme.base", "dark")
        st._config.set_option("theme.primaryColor", "#3b82f6")
        st._config.set_option("theme.backgroundColor", "#020617")
        st._config.set_option("theme.secondaryBackgroundColor", "#111827")
        st._config.set_option("theme.textColor", "#e5e7eb")
    else:
        st._config.set_option("theme.base", "light")
        st._config.set_option("theme.primaryColor", "#2563eb")
        st._config.set_option("theme.backgroundColor", "#ffffff")
        st._config.set_option("theme.secondaryBackgroundColor", "#f1f5f9")
        st._config.set_option("theme.textColor", "#020617")

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"

# Toggle switch instead of radio button
is_dark = st.toggle("🌙 Dark mode", value=(st.session_state.theme_choice == "Dark"))
new_theme = "Dark" if is_dark else "Light"

if new_theme != st.session_state.theme_choice:
    st.session_state.theme_choice = new_theme
    apply_theme(new_theme)
    st.rerun()

apply_theme(st.session_state.theme_choice)


# =======================
# CSS (cards + animations only)
# =======================
st.markdown("""
<style>
.main-title {
    font-size: 44px; font-weight: 800; text-align: center; margin-bottom: 10px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.card { border-radius: 18px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
.price-box { font-size: 32px; font-weight: 700; color: #22c55e; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)} }
</style>
""", unsafe_allow_html=True)

# =======================
# Title
# =======================
st.markdown("<div class='main-title'>🚗 AI Car Price Predictor</div>", unsafe_allow_html=True)
st.caption("Single prediction • Multi‑car comparison • Confidence interval • Charts")

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipe = load_model()

# =======================
# Load Dataset for dropdowns
# =======================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data_df = load_data()

companies = sorted(data_df['company'].dropna().unique().tolist())
models_by_company = data_df.groupby('company')['name'].unique().to_dict()

# =======================
# Prediction Helpers
# =======================

def predict_price(row):
    df = pd.DataFrame([row], columns=['name','company','year','kms_driven','fuel_type'])
    return float(pipe.predict(df)[0])


def confidence_interval(price, pct=0.08):
    delta = price * pct
    return price - delta, price + delta


def car_form(key):
    company = st.selectbox("Company", companies, key=f"comp_{key}")
    model_list = sorted(models_by_company.get(company, []))
    name = st.selectbox("Model", model_list, key=f"name_{key}")
    year = st.number_input("Year", 1995, 2026, 2019, key=f"year_{key}")
    kms = st.number_input("Kilometers Driven", 0, 500000, 20000, key=f"kms_{key}")
    fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric"], key=f"fuel_{key}")
    return [name, company, year, kms, fuel]

# =======================
# Tabs
# =======================
tab1, tab2 = st.tabs(["🔮 Predict", "⚖ Compare (Multi‑Car)"])

# =======================
# Single Prediction
# =======================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    car = car_form("single")

    if st.button("Predict Price 🚀"):
        price = predict_price(car)
        low, high = confidence_interval(price)
        st.markdown(f"<div class='price-box'>₹ {price:,.0f}</div>", unsafe_allow_html=True)
        st.info(f"Confidence range: ₹ {low:,.0f}  –  ₹ {high:,.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

# =======================
# Multi‑Car Comparison
# =======================
with tab2:
    st.subheader("Compare 3–5 Cars")

    count = st.slider("Number of cars", 3, 5, 3)

    cars = []
    cols = st.columns(count)

    for i in range(count):
        with cols[i]:
            st.markdown(f"### Car {i+1}")
            cars.append(car_form(f"multi_{i}"))

    if st.button("Compare Prices 📊"):
        prices = []
        labels = []

        for i, c in enumerate(cars):
            p = predict_price(c)
            prices.append(p)
            labels.append(f"Car {i+1}")

        result_df = pd.DataFrame({"Car": labels, "Predicted Price": prices})
        st.dataframe(result_df, use_container_width=True)

        st.markdown("### Price Comparison Chart")
        chart_df = pd.DataFrame({"Price": prices}, index=labels)
        st.bar_chart(chart_df)

# =======================
# Deployment Notes
# =======================
st.markdown("---")
st.markdown("""
© Developed by Aaditya Mathur
**GitHub:** https://github.com/adityamathur456
""")
