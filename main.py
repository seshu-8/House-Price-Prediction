"""
House Price Prediction - Main Entry Point
==========================================
Run this file to execute the full pipeline:
  python main.py
"""

from src.data_loader import load_or_generate_data
from src.preprocessing import preprocess_data
from src.eda import run_eda
from src.model import train_models, compare_models
from src.predict import predict_price

def main():
    print("=" * 60)
    print("   HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 60)

    # Step 1: Load / Generate Dataset
    print("\n[1/5] Loading dataset...")
    df = load_or_generate_data()
    print(f"      Dataset shape: {df.shape}")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Step 3: EDA
    print("\n[3/5] Running EDA (saving charts to outputs/)...")
    run_eda(df)

    # Step 4: Train & Evaluate Models
    print("\n[4/5] Training models...")
    models = train_models(X_train, y_train, X_test, y_test)

    # Step 5: Compare Models
    print("\n[5/5] Model comparison:")
    compare_models(models, X_test, y_test)

    # Demo Prediction
    print("\n" + "=" * 60)
    print("   SAMPLE PREDICTION")
    print("=" * 60)
    sample = {
        "area_sqft": 1800,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5,
        "parking": 1,
        "location_score": 7,
        "furnishing": "semi-furnished",
        "property_type": "apartment"
    }
    predict_price(models["Random Forest"], sample)

if __name__ == "__main__":
    main()
