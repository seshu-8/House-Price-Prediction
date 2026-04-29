"""
model.py
========
Trains and evaluates three regression models:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor

Outputs:
  - Evaluation table (MAE, RMSE, R²)
  - Actual vs Predicted plot (outputs/)
  - Feature importance chart (outputs/)
  - Saved model files (models/)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Train all models ────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test) -> dict:
    definitions = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                                    random_state=42, n_jobs=-1),
    }

    trained = {}
    for name, model in definitions.items():
        model.fit(X_train, y_train)
        trained[name] = model
        joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")
        print(f"      ✓ {name} trained")

    return trained


# ── Evaluate + Compare ──────────────────────────────────────────────────────
def compare_models(models: dict, X_test, y_test):
    records = []
    preds_store = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        preds_store[name] = preds

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        records.append({"Model": name, "MAE": round(mae, 2),
                         "RMSE": round(rmse, 2), "R²": round(r2, 4)})

    results_df = pd.DataFrame(records).sort_values("R²", ascending=False)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("outputs/model_comparison.csv", index=False)

    # Actual vs Predicted (best model = Random Forest)
    _plot_actual_vs_predicted(y_test, preds_store["Random Forest"])

    # Feature importance
    rf_model = models["Random Forest"]
    _plot_feature_importance(rf_model)

    # Bar chart comparison
    _plot_model_comparison(results_df)


# ── Actual vs Predicted ─────────────────────────────────────────────────────
def _plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5, s=20, color="#4C72B0", label="Predictions")
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
    plt.xlabel("Actual Price (Lakhs)")
    plt.ylabel("Predicted Price (Lakhs)")
    plt.title("Actual vs Predicted Price (Random Forest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/06_actual_vs_predicted.png", dpi=150)
    plt.close()


# ── Feature Importance ──────────────────────────────────────────────────────
def _plot_feature_importance(rf_model):
    feature_names = [
        "area_sqft", "bedrooms", "bathrooms", "age_years",
        "parking", "location_score", "furnishing_enc",
        "property_type_enc", "room_ratio", "value_index"
    ]
    importance = rf_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df.sort_values("Importance", ascending=True, inplace=True)

    plt.figure(figsize=(8, 5))
    plt.barh(fi_df["Feature"], fi_df["Importance"], color="#4C72B0")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig("outputs/07_feature_importance.png", dpi=150)
    plt.close()


# ── Model Comparison Bar Chart ──────────────────────────────────────────────
def _plot_model_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["MAE", "RMSE", "R²"]
    colors  = ["#DD8452", "#55A868", "#4C72B0"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(results_df["Model"], results_df[metric], color=color, edgecolor="white")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/08_model_comparison.png", dpi=150)
    plt.close()
