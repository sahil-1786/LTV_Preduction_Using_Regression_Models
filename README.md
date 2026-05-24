# E-Commerce Customer Lifetime Value (LTV) Prediction

A comprehensive machine learning regression pipeline designed to predict customer **Customer Lifetime Value (LTV)** using e-commerce behavioral metrics. The project explores multiple modeling paradigms—from classical regularized linear regressions to non-linear ensemble methods—while implementing rigorous feature engineering, outlier mitigation, and multicollinearity auditing.

---

## 🚀 Key Features & Pipeline Architecture

* **Robust Data Quality Pipeline:** Automates median imputation for skewed numerical variables and mode imputation for categorical fields across a 50,000-row dataset.
* **Multicollinearity Auditing:** Computes Variance Inflation Factors (VIF) to detect and isolate unstable feature dependencies.
* **Target & Feature Engineering:** Corrects right-skewed variables using Yeo-Johnson transformations, applies log-scaling to the target vector to ensure homoscedasticity, and performs extreme value Winsorization.
* **Multi-Model Benchmark Suite:** Evaluates and cross-validates 5 distinct regression strategies under identical 10-fold validation conditions.
* **Automated Analytics Engine:** Outputs a standard comprehensive comparison matrix alongside 11 professional visualizations detailing model residuals, feature importance, and error metrics.

---

## 🛠️ Feature Engineering Matrix

The script transforms raw behavioral data into highly predictive structural indicators:

| Engineered Feature | Formulation | Business Rationale |
| :--- | :--- | :--- |
| **Engagement Index** | Session Duration Avg × Pages Per Session | Quantifies the total active processing depth per user visit. |
| **Purchase Recency** | 1 / (Days Since Last Purchase + 1) | Standardizes recency to a bounded (0, 1] scale where higher means active. |
| **Loyalty Ratio** | Total Purchases / (Membership Years + 1) | Measures purchase velocity relative to historical tenure. |

---

## 📊 Model Performance Benchmark

After applying targeted distributions corrections (Log-Transform, Yeo-Johnson, and Standardization), the models produced the following diagnostic metrics under 10-fold Cross-Validation:

| Model Hierarchy | R² Score | Adj. R² | RMSE | MAE | Mean CV R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **0.9766** | **0.9765** | **0.1015** | **0.0693** | **0.9489** |
| **Ridge Regression** | 0.9138 | 0.9137 | 0.1948 | 0.1301 | 0.9137 |
| **OLS Regression (Transformed)** | 0.9138 | 0.9138 | 0.1948 | 0.1301 | 0.9137 |
| **Principal Component Regression (PCR)** | 0.5213 | 0.5212 | 0.4590 | 0.3555 | 0.5211 |
| **OLS Regression (Original Base)** | 0.5052 | 0.5049 | 638.1736 | 444.6300 | 0.5028 |

### 🔍 Crucial Statistical Insights

1. **The Transformation Leap:** Transforming the baseline target variable from highly skewed (1.44) to balanced via a Log-Transform immediately boosted linear model explanatory power ($R^2$) from **50.5% to 91.3%**.
2. **Top Predictors:** Statistical testing confirms that **Total Purchases**, **Average Order Value**, **Loyalty Ratio**, and **Engagement Index** hold the strongest absolute T-statistic weight on a customer's long-term valuation.
3. **Multicollinearity:** The engineered `Engagement_Index` exhibits high collinearity (VIF ≈ 20.08). While this destabilizes standard OLS coefficients, **Ridge Regression** successfully regularizes the weights without degrading prediction accuracy.

---

## 📁 Saved Production Deliverables

All generated outputs are piped automatically directly to `/mnt/user-data/outputs/`:

* `model_metrics_comparison.csv` — Full benchmark table containing evaluation data.
* `00_vif_multicollinearity.png` — Feature dependency visualization.
* `02_r2_comparison.png` & `03_error_metrics_comparison.png` — Structural model comparisons.
* `05_predicted_vs_actual.png` & `06_residual_distributions.png` — Model fit and normality analysis.
* `08_feature_importance.png` — Machine learning driver weights plot.

---

## 💻 Requirements & Dependencies

The baseline script runs using modern data science libraries:

```text
python >= 3.8
numpy
pandas
scikit-learn
scipy
statsmodels
matplotlib
seaborn
