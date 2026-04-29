# 🎤 Interview Preparation — House Price Prediction

## Core ML Questions

**Q1: Why Random Forest over Linear Regression here?**
> Linear Regression assumes a linear relationship. House prices depend on non-linear interactions (e.g., a 5BHK villa in prime area ≠ just sum of features). Random Forest handles non-linearity, outliers, and feature interactions automatically.

**Q2: What is R² score? What does 0.97 mean?**
> R² (coefficient of determination) measures how much variance in the target the model explains. R²=0.97 means the model explains 97% of price variation — very strong for a regression problem.

**Q3: Why did you use StandardScaler?**
> Features had different scales (area in thousands vs bedrooms in 1-5). Without scaling, models like Linear Regression give disproportionate weight to large-scale features. StandardScaler normalizes to mean=0, std=1.

**Q4: What is MAE vs RMSE?**
> MAE = average absolute error (less sensitive to outliers)
> RMSE = root mean square error (penalizes large errors more)
> Use RMSE when large errors are more costly.

**Q5: How did you handle categorical features?**
> Used LabelEncoder for ordinal features (furnishing: unfurnished < semi < furnished) and for property type. For high-cardinality features, OneHotEncoder would be better.

**Q6: What is overfitting? How did you check it?**
> When model memorizes training data but fails on test data. Checked by comparing train R² vs test R² — if train >> test, model is overfitting. Random Forest with max_depth=12 prevents this.

**Q7: Why feature engineering?**
> Raw features don't always capture relationships well.
> - `room_ratio` = bathrooms/bedrooms captures comfort relative to size
> - `value_index` = location_score × area captures premium location value

---

## Business Analyst Questions

**Q: How would a bank use this model?**
> For loan approval: if borrower claims property is worth ₹150L but model predicts ₹90L, the bank flags it for manual review. Reduces over-lending risk.

**Q: What KPIs would you track for this model in production?**
> - Prediction accuracy (MAPE — mean absolute percentage error)
> - Model drift (accuracy degrading as market changes)
> - Outlier rate (% of predictions > 2× standard deviations off)

**Q: How would you improve this model?**
> 1. Add real location data (lat/long or pin codes)
> 2. Include nearby amenities (school, hospital, metro distance)
> 3. Add time-series component (price trends over months)
> 4. Use XGBoost or LightGBM for higher accuracy

---

## Quick Stats to Memorize
- Dataset: 1,000 rows, 8 features, 1 target
- Best model: Random Forest (200 trees, max_depth=12)
- R² = ~0.97, RMSE = ~7.3 Lakhs
- Feature engineering: room_ratio, value_index
- No external API or real data needed — 100% synthetic
