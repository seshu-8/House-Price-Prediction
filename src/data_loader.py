"""
data_loader.py
==============
Generates a realistic synthetic housing dataset.
No external CSV needed - works out of the box.
"""

import pandas as pd
import numpy as np
import os

def load_or_generate_data(n_samples: int = 1000, save: bool = True) -> pd.DataFrame:
    """
    Generate a synthetic housing dataset with realistic price patterns.

    Features:
        area_sqft        - Size of house in sq ft
        bedrooms         - Number of bedrooms
        bathrooms        - Number of bathrooms
        age_years        - Age of property
        parking          - Parking spots (0/1/2)
        location_score   - Location rating 1-10 (higher = prime area)
        furnishing       - unfurnished / semi-furnished / furnished
        property_type    - apartment / villa / independent house

    Target:
        price_lakhs      - Price in lakhs (Indian real estate scale)
    """
    np.random.seed(42)

    area_sqft       = np.random.randint(500, 5000, n_samples)
    bedrooms        = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.35, 0.15, 0.1])
    bathrooms       = np.clip(bedrooms - 1 + np.random.randint(0, 2, n_samples), 1, 5)
    age_years       = np.random.randint(0, 30, n_samples)
    parking         = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2])
    location_score  = np.random.randint(1, 11, n_samples)
    furnishing      = np.random.choice(
        ["unfurnished", "semi-furnished", "furnished"],
        n_samples, p=[0.3, 0.4, 0.3]
    )
    property_type   = np.random.choice(
        ["apartment", "villa", "independent house"],
        n_samples, p=[0.55, 0.2, 0.25]
    )

    # Furnishing multiplier
    furnish_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    furn_val = np.array([furnish_map[f] for f in furnishing])

    # Property type multiplier
    prop_map = {"apartment": 0, "independent house": 1, "villa": 2}
    prop_val = np.array([prop_map[p] for p in property_type])

    # Price formula (realistic India-scale)
    base_price = (
        area_sqft * 0.03
        + bedrooms * 5
        + bathrooms * 3
        + location_score * 8
        + parking * 4
        + furn_val * 6
        + prop_val * 12
        - age_years * 0.5
    )

    noise = np.random.normal(0, 8, n_samples)
    price_lakhs = np.clip(base_price + noise, 10, 500).round(2)

    df = pd.DataFrame({
        "area_sqft":      area_sqft,
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "age_years":      age_years,
        "parking":        parking,
        "location_score": location_score,
        "furnishing":     furnishing,
        "property_type":  property_type,
        "price_lakhs":    price_lakhs
    })

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/housing_data.csv", index=False)
        print(f"      Saved: data/housing_data.csv")

    return df
