# 🏠 House Price Prediction using Regression Models

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview

An end-to-end Machine Learning project that predicts residential property prices based on features like area, location, bedrooms, furnishing type, and property age.

Built as a **portfolio project** targeting Data Science / ML / Data Analyst roles.

---

## 🔍 Problem Statement

> Given a set of property attributes, predict the **price in Lakhs (₹)** with the highest accuracy possible.

Real estate companies, banks, and property portals use price prediction to:
- Generate automated valuations for listings
- Assess loan eligibility based on property value
- Identify undervalued investment opportunities
- Power buyer recommendation engines

---

## 🏗️ Architecture

```
Raw Property Data
      │
      ▼
┌─────────────────────┐
│   Data Cleaning     │  → Remove duplicates, fill nulls
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Engineering│  → room_ratio, value_index, encoding
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Scaling    │  → StandardScaler
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Regression Models │  → Linear / Decision Tree / Random Forest
└─────────┬───────────┘
          │
          ▼
    Predicted Price (₹ Lakhs)
```

---

## 🛠️ Tech Stack

| Category     | Tools                                      |
|-------------|---------------------------------------------|
| Language     | Python 3.13                                |
| Data         | Pandas, NumPy                              |
| Visualization| Matplotlib, Seaborn                        |
| Models       | Scikit-learn (LR, DT, RF)                  |
| Serialization| Joblib                                     |
| Notebook     | Jupyter                                    |

---

## 📁 Folder Structure

```
House-Price-Prediction/
│
├── data/                  # Generated CSV dataset
├── notebooks/             # Jupyter notebook (full walkthrough)
├── src/                   # Python modules
│   ├── data_loader.py     # Synthetic data generation
│   ├── preprocessing.py   # Cleaning, encoding, scaling
│   ├── eda.py             # EDA charts
│   ├── model.py           # Training + evaluation
│   └── predict.py         # Single property prediction
├── models/                # Saved .pkl model files
├── outputs/               # Charts and evaluation results
├── images/                # Screenshots for README
├── main.py                # Entry point (run this)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python main.py
```

### 5. Or open the notebook
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

---

## 📊 Dataset

- **Type:** Synthetic (programmatically generated, no API needed)
- **Rows:** 1,000 properties
- **Features:** 8 input features + 1 target

| Feature        | Type        | Description                |
|---------------|-------------|----------------------------|
| area_sqft      | Numeric     | Total area in square feet  |
| bedrooms       | Numeric     | Number of bedrooms         |
| bathrooms      | Numeric     | Number of bathrooms        |
| age_years      | Numeric     | Age of property            |
| parking        | Numeric     | Parking spots (0/1/2)      |
| location_score | Numeric     | Prime area rating (1-10)   |
| furnishing     | Categorical | unfurnished/semi/furnished |
| property_type  | Categorical | apartment/villa/house      |
| **price_lakhs**| **Target**  | **Price in ₹ Lakhs**       |

---

## 🤖 Models & Results

| Model             | MAE   | RMSE  | R²     |
|------------------|-------|-------|--------|
| Linear Regression | ~8.5  | ~11.2 | ~0.93  |
| Decision Tree     | ~6.2  | ~8.9  | ~0.96  |
| **Random Forest** | **~5.1** | **~7.3** | **~0.97** |

> Random Forest gives the best accuracy and is used for final predictions.

---

## 📈 Output Charts

1. `01_price_area_distribution.png` — Distribution of price and area
2. `02_correlation_heatmap.png` — Feature correlation matrix
3. `03_price_vs_features.png` — Scatter plots (price vs key features)
4. `04_price_by_category.png` — Box plots by furnishing / property type
5. `05_pairplot.png` — Pairplot of key features
6. `06_actual_vs_predicted.png` — Model accuracy visual
7. `07_feature_importance.png` — Random Forest feature ranking
8. `08_model_comparison.png` — MAE / RMSE / R² comparison bar chart

---

## 🧪 Sample Prediction

```python
sample = {
    "area_sqft":      1800,
    "bedrooms":       3,
    "bathrooms":      2,
    "age_years":      5,
    "parking":        1,
    "location_score": 7,
    "furnishing":     "semi-furnished",
    "property_type":  "apartment"
}
# Output: Predicted Price: ₹112.47 Lakhs
```

---

## 🎓 Learning Outcomes

- Synthetic data generation for real-world simulation
- End-to-end ML pipeline (EDA → Model → Deployment)
- Regression model comparison and selection
- Feature engineering (room_ratio, value_index)
- Model persistence using joblib
- Production-ready folder structure

---

## 📬 Connect

Built by **Seshu** | GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

> ⭐ Star this repo if it helped you!
