# 🚗 Car Price Prediction with Machine Learning

An end-to-end Machine Learning regression solution designed to accurately predict the market selling price of used cars based on mechanical specifications, vehicle age, mileage, fuel type, transmission, and brand reputation.

---

## 📌 Project Overview

Valuing used cars accurately is critical for peer-to-peer automotive marketplaces, dealerships, insurance adjusters, and prospective buyers. This project implements a complete data science workflow:
* **Data Sourcing & Audit:** Sourced the Kaggle CarDekho Used Car dataset (v3) containing 8,128 vehicle listings.
* **Data Cleaning & Sanitization:** Removed 1,202 duplicate listings, cleaned corrupted entries (such as empty `' bhp'` strings and zero mileage), extracted numerical floats from measurement units (`kmpl`, `km/kg`, `CC`, `bhp`), and standardized inconsistent categorical casing.
* **Feature Engineering:** Derived vehicle age (`car_age = 2024 - year`), extracted manufacturer (`brand`) from the listing name, resolved multi-word brands (e.g. `'Land Rover'`), and grouped low-frequency brands.
* **Exploratory Data Analysis (EDA):** Conducted price distribution analysis (raw and log-transformed), fuel-type box plots, depreciation curve scatter plots, and correlation heatmaps.
* **Preprocessing Pipeline:** Encapsulated numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder(drop='first')`) into a scikit-learn `ColumnTransformer` to guarantee zero data leakage.
* **Model Benchmarking:** Trained and evaluated three distinct regression models: **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**.
* **Metrics:** Evaluated models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).
* **Feature Importance:** Analyzed key price drivers using tree split variance reduction.
* **Inference Demo:** Deployed the trained model pipeline for instant prediction on unseen vehicle inputs.

---

## 📁 Repository Structure

```text
car_price_prediction/
├── data/
│   ├── raw/
│   │   └── car_details_v3.csv          # Raw CarDekho dataset (8,128 records)
│   └── processed/
│       └── cleaned_car_details.csv     # Cleaned, feature-engineered dataset (6,702 records)
├── models/
│   └── best_car_price_model.joblib     # Exported Gradient Boosting model pipeline
├── notebooks/
│   └── car_price_prediction.ipynb      # Fully executed, commented Jupyter Notebook
├── reports/
│   └── figures/
│       ├── selling_price_distribution.png
│       ├── price_vs_fuel_boxplot.png
│       ├── price_vs_car_age_scatter.png
│       ├── correlation_heatmap.png
│       ├── top_brands_price_barplot.png
│       ├── feature_importance.png
│       └── actual_vs_predicted.png
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                # Data cleaning and feature engineering
│   ├── eda.py                          # Exploratory data analysis & figure generator
│   ├── train.py                        # Model training, benchmarking & serialization
│   ├── inference.py                    # Sample inference script on unseen inputs
│   ├── build_notebook.py               # Notebook builder script
│   └── render_notebook.py              # In-process execution & cell output renderer
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & results summary
```

---

## 📊 Model Performance Benchmark

Models were trained on **80% of the dataset (5,361 listings)** and tested on **20% unseen data (1,341 listings)** with a fixed random state (`random_state=42`).

| Regression Model | Test $R^2$ Score | Test MAE (₹) | Test RMSE (₹) | Train $R^2$ Score | Train MAE (₹) | Train RMSE (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** 🏆 | **0.9300** | **₹70,462** | **₹119,376** | **0.9797** | **₹54,890** | **₹76,785** |
| **Random Forest Regressor** | 0.9186 | ₹72,119 | ₹128,741 | 0.9773 | ₹40,768 | ₹81,231 |
| **Linear Regression** | 0.7431 | ₹135,022 | ₹228,677 | 0.7323 | ₹143,174 | ₹278,744 |

### 🔍 Key Performance Insights
1. **Ensemble Superiority:** Both **Gradient Boosting ($R^2 = 93.0\%$)** and **Random Forest ($R^2 = 91.9\%$)** drastically outperform baseline **Linear Regression ($R^2 = 74.3\%$)**, confirming that vehicle pricing exhibits non-linear relationships and interaction effects across power, brand, and age.
2. **Error Margins:** The Gradient Boosting model predicts car market values with an average error of just **~₹70,462** on test cars ranging from ₹30,000 to over ₹40,00,000.

---

## 💡 Key Price Determinants (Feature Importance)

Extracted from the top-performing Gradient Boosting Regressor:
1. **Engine Max Power (`max_power` - 62.0%):** The single strongest pricing factor. Horsepower directly separates economy hatchbacks from luxury sedans and performance vehicles.
2. **Vehicle Age (`car_age` - 23.7%):** Captures ongoing depreciation. Cars lose significant valuation during their first 3–7 years.
3. **Odometer Reading (`km_driven` - 4.6%):** Wear-and-tear accumulation.
4. **Engine Displacement (`engine` - 2.6%):** Correlates with vehicle class and tax/market category.
5. **Fuel Economy (`mileage` - 2.6%):** High mileage economy cars inhabit lower pricing brackets.
6. **Brand, Transmission & Fuel Type (~4.5%):** Automatic transmissions and diesel variants command notable premiums in the resale market.

---

## 🚀 Getting Started & Execution

### 1. Requirements & Environment
Ensure Python 3.10+ is installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing Pipeline
Cleans the raw dataset, extracts numeric specs, and generates `data/processed/cleaned_car_details.csv`:
```bash
python3 src/data_pipeline.py
```

### 3. Generate Exploratory Visualizations
Produces high-resolution charts saved in `reports/figures/`:
```bash
python3 src/eda.py
```

### 4. Train and Evaluate Models
Trains all candidate models, evaluates test metrics, saves the best model to `models/best_car_price_model.joblib`, and outputs the benchmark table:
```bash
python3 src/train.py
```

### 5. Run Inference on Test Vehicles
Predicts prices on unseen test samples:
```bash
python3 src/inference.py
```

### 6. Open the Jupyter Notebook
Launch Jupyter to interactively inspect or re-run the complete workflow:
```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```
*(The notebook is already pre-rendered with all tables, charts, and cell outputs.)*
