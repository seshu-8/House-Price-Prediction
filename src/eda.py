"""
eda.py
======
Exploratory Data Analysis:
  - Distribution plots
  - Correlation heatmap
  - Price vs key features
  - Box plots by category
All charts are saved to outputs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")


def run_eda(df: pd.DataFrame):
    _price_distribution(df)
    _correlation_heatmap(df)
    _price_vs_area(df)
    _price_by_category(df)
    _pairplot(df)
    print("      Charts saved in outputs/")


# ── 1. Price Distribution ───────────────────────────────────────────────────
def _price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["price_lakhs"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("House Price Distribution")
    axes[0].set_xlabel("Price (Lakhs)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["area_sqft"], bins=40, color="#DD8452", edgecolor="white")
    axes[1].set_title("Area Distribution")
    axes[1].set_xlabel("Area (sq ft)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("outputs/01_price_area_distribution.png", dpi=150)
    plt.close()


# ── 2. Correlation Heatmap ─────────────────────────────────────────────────
def _correlation_heatmap(df):
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, square=True
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
    plt.close()


# ── 3. Price vs Area (scatter) ─────────────────────────────────────────────
def _price_vs_area(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pairs = [
        ("area_sqft", "Area (sq ft)"),
        ("location_score", "Location Score"),
        ("age_years", "Age (years)"),
    ]

    for ax, (col, label) in zip(axes, pairs):
        ax.scatter(df[col], df["price_lakhs"], alpha=0.4, s=15, color="#4C72B0")
        ax.set_xlabel(label)
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title(f"Price vs {label}")

    plt.tight_layout()
    plt.savefig("outputs/03_price_vs_features.png", dpi=150)
    plt.close()


# ── 4. Box plots by category ──────────────────────────────────────────────
def _price_by_category(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="furnishing", y="price_lakhs", ax=axes[0],
                order=["unfurnished", "semi-furnished", "furnished"],
                hue="furnishing", legend=False, palette="pastel")
    axes[0].set_title("Price by Furnishing")
    axes[0].set_xlabel("Furnishing Type")
    axes[0].set_ylabel("Price (Lakhs)")

    sns.boxplot(data=df, x="property_type", y="price_lakhs", ax=axes[1],
                hue="property_type", legend=False, palette="pastel")
    axes[1].set_title("Price by Property Type")
    axes[1].set_xlabel("Property Type")
    axes[1].set_ylabel("Price (Lakhs)")

    plt.tight_layout()
    plt.savefig("outputs/04_price_by_category.png", dpi=150)
    plt.close()


# ── 5. Pairplot (key features) ────────────────────────────────────────────
def _pairplot(df):
    cols = ["area_sqft", "bedrooms", "location_score", "age_years", "price_lakhs"]
    pair_df = df[cols].sample(300, random_state=42)   # sample for speed

    g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20})
    g.fig.suptitle("Pairplot of Key Features", y=1.01)
    g.savefig("outputs/05_pairplot.png", dpi=100)
    plt.close()
