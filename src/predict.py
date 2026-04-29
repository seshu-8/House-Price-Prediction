"""
predict.py
==========
Handles prediction for a single new property input.
Can also be used standalone:
    python src/predict.py
"""

import numpy as np
import joblib
import os


FURNISH_MAP = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
PROP_MAP    = {"apartment": 0, "independent house": 1, "villa": 2}


def _encode_input(sample: dict) -> np.ndarray:
    """Convert raw property dict → scaled feature vector."""
    furn_enc = FURNISH_MAP.get(sample["furnishing"].lower(), 1)
    prop_enc = PROP_MAP.get(sample["property_type"].lower(), 0)

    bedrooms  = sample["bedrooms"]
    bathrooms = sample["bathrooms"]
    area      = sample["area_sqft"]
    loc       = sample["location_score"]

    room_ratio  = bathrooms / bedrooms if bedrooms > 0 else 1
    value_index = loc * area / 1000

    raw_features = np.array([[
        area,
        bedrooms,
        bathrooms,
        sample["age_years"],
        sample["parking"],
        loc,
        furn_enc,
        prop_enc,
        room_ratio,
        value_index
    ]])

    # Scale using saved scaler
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler.transform(raw_features)
    return raw_features


def predict_price(model, sample: dict):
    """
    Predict price for a property dict.

    Args:
        model  : trained sklearn model
        sample : dict with property details

    Returns:
        float  : predicted price in lakhs
    """
    X = _encode_input(sample)
    price = model.predict(X)[0]

    print(f"\n  Property Details:")
    for k, v in sample.items():
        print(f"    {k:20s}: {v}")
    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  Predicted Price: ₹{price:.2f} Lakhs  ║")
    print(f"  ╚══════════════════════════════════╝")
    return price


# ── Standalone usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    model = joblib.load("models/Random_Forest.pkl")

    custom_property = {
        "area_sqft":      2200,
        "bedrooms":       3,
        "bathrooms":      2,
        "age_years":      3,
        "parking":        1,
        "location_score": 8,
        "furnishing":     "furnished",
        "property_type":  "villa"
    }

    predict_price(model, custom_property)
