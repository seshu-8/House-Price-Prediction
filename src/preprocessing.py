"""
preprocessing.py
================
Handles:
  - Missing value treatment
  - Label encoding for categorical features
  - Feature scaling
  - Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # ── 1. Drop duplicates ─────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"      Removed {before - len(df)} duplicate rows")

    # ── 2. Handle missing values (Python 3.13 / Pandas 2.x compatible) ──
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── 3. Encode categoricals ─────────────────────────────────────────
    le_furnishing = LabelEncoder()
    le_proptype   = LabelEncoder()

    df["furnishing_enc"]    = le_furnishing.fit_transform(df["furnishing"])
    df["property_type_enc"] = le_proptype.fit_transform(df["property_type"])

    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(le_furnishing, "models/le_furnishing.pkl")
    joblib.dump(le_proptype,   "models/le_proptype.pkl")

    # ── 4. Feature engineering ─────────────────────────────────────────
    df["price_per_sqft"] = (df["price_lakhs"] / df["area_sqft"]).round(4)
    df["room_ratio"]     = (df["bathrooms"] / df["bedrooms"]).round(3)
    df["value_index"]    = (df["location_score"] * df["area_sqft"] / 1000).round(3)

    # ── 5. Feature / Target split ──────────────────────────────────────
    feature_cols = [
        "area_sqft", "bedrooms", "bathrooms", "age_years",
        "parking", "location_score", "furnishing_enc",
        "property_type_enc", "room_ratio", "value_index"
    ]

    X = df[feature_cols]
    y = df["price_lakhs"]

    # ── 6. Scale features ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")

    # ── 7. Train-test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, feature_cols
