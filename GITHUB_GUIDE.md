# 📤 GitHub Upload Guide

## Step 1 — Create Repository on GitHub
1. Go to github.com → New Repository
2. Name: `House-Price-Prediction`
3. Description: `End-to-end ML project predicting house prices using Linear Regression, Decision Tree & Random Forest | Python 3.13`
4. Set: Public
5. Do NOT initialize with README (you have one)
6. Click: Create Repository

---

## Step 2 — Initialize Git Locally
```bash
cd House-Price-Prediction
git init
git add .
git commit -m "Initial commit: House Price Prediction project"
```

---

## Step 3 — Connect & Push
```bash
git remote add origin https://github.com/YOUR_USERNAME/House-Price-Prediction.git
git branch -M main
git push -u origin main
```

---

## Step 4 — Day-wise Commit Plan

| Day | What to Commit | Commit Message |
|-----|----------------|----------------|
| 1   | requirements.txt, folder structure | `setup: project structure and dependencies` |
| 2   | data_loader.py, housing_data.csv | `data: synthetic dataset generation` |
| 3   | eda.py + EDA charts | `eda: exploratory data analysis and charts` |
| 4   | preprocessing.py, model.py | `model: preprocessing pipeline and model training` |
| 5   | outputs/model_comparison.csv | `eval: model evaluation results (R²=0.97)` |
| 6   | All outputs/ charts | `viz: actual vs predicted, feature importance charts` |
| 7   | README.md, screenshots | `docs: complete README with results and screenshots` |

---

## Best GitHub Tags (Topics)
```
machine-learning, regression, house-price-prediction, scikit-learn,
random-forest, python, data-science, pandas, matplotlib, portfolio-project
```

---

## Screenshots to Capture Before Upload

1. Terminal output of `python main.py`
2. outputs/02_correlation_heatmap.png
3. outputs/06_actual_vs_predicted.png
4. outputs/07_feature_importance.png
5. outputs/08_model_comparison.png
6. GitHub repository page (after upload)

Save all in: `images/` folder → add to README.md
