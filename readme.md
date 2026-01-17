# 🚗 Car Price Prediction System

### Streamlit + CatBoost • Production‑Ready ML Web Application

A modern, production‑ready web application for predicting and comparing used‑car prices using a trained **CatBoost regression model**. The system supports single‑car prediction, multi‑car comparison, confidence intervals, a polished UI with dark/light mode, and deployment via Docker or Streamlit Cloud.

---

## 👨‍💻 Author

**Aaditya Mathur**\
GitHub: [https://github.com/adityamathur456](https://github.com/adityamathur456)

---

## ✨ Key Features

- AI‑based car price prediction using CatBoost
- Company & model dropdowns auto‑loaded from the dataset
- Multi‑car comparison (3–5 cars)
- Animated price comparison bar chart
- Prediction confidence interval
- Dark / light theme toggle (native Streamlit)
- Fast caching with Streamlit
- Docker‑ready deployment
- Streamlit Cloud ready

---

## 📁 Project Structure

```
car-price-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   └── Cleaned_Car_data.csv
│   └── raw/
│       └── quikr_car.csv
│
├── models/
│   ├── catboost_model.pkl
│   ├── catboost_v1.pkl
│   └── linear_regression_v1.pkl
│
├── model_training/
│   └── model_train.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Python 3.10** (recommended for CatBoost compatibility)
- Streamlit (UI framework)
- CatBoost (ML model)
- Scikit‑learn (preprocessing & pipeline)
- Pandas / NumPy (data handling)
- Matplotlib / Seaborn (visualization)
- Docker (containerization)

---

## 📦 Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
cd car-price-prediction
streamlit run app/app.py
```

Then open in your browser:

```
http://localhost:8501
```

---

## 🐳 Run with Docker

### Build the image

```bash
docker build -t car-price-predict-model .
```

### Run the container

```bash
docker run -p 8501:8501 car-price-predict-model
```

Open in your browser:

```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to GitHub
2. Visit: [[car-price-predictor](https://car-price-predictor-comparator-fy28vojgx328hx8asrkgcj.streamlit.app/)]
3. Select your repository
4. Set the main file path to:

```
app/app.py
```

5. Click **Deploy**

---

## 🧠 Model Input Format

The model expects the following features:

| Feature     | Type   |
| ----------- | ------ |
| name        | string |
| company     | string |
| year        | int    |
| kms\_driven | int    |
| fuel\_type  | string |

---

## 📊 Sample Prediction Code

```python
pipe.predict(pd.DataFrame(
    columns=["name", "company", "year", "kms_driven", "fuel_type"],
    data=[["Maruti Suzuki Swift", "Maruti", 2019, 100, "Petrol"]]
))
```

---

## 🔒 Version Compatibility (Important)

To ensure the saved model loads correctly, use the following versions:

- `numpy == 1.26.4`
- `scikit-learn == 1.3.2`
- `catboost == 1.2.5`
- `python == 3.10` (recommended)

> Note: Using different versions of scikit‑learn and catboost may require re‑saving or retraining the model.

---

## 📝 License

This project is intended for educational, academic, and internship use.

---

## 📬 Contact

For queries or collaboration:

**Aaditya Mathur**\
GitHub: [https://github.com/adityamathur456](https://github.com/adityamathur456)

---

⭐ If you find this project useful, consider giving it a star on GitHub.

