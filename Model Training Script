"""
Model Training and Evaluation Pipeline for Car Price Prediction.
Trains Linear Regression, Random Forest, and Gradient Boosting models,
evaluates using MAE, RMSE, and R2 score, and plots feature importance.
"""

import os
import joblib
import numpy as np
import pandas as pd

os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def build_preprocessor(numeric_features, categorical_features):
    """Create ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )


def train_and_evaluate(data_path: str, models_dir: str, reports_dir: str):
    """Main training, evaluation, and visualization routine."""
    os.makedirs(models_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset for modeling: {df.shape}")

    # Define feature groups
    numeric_features = ['car_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    categorical_features = ['brand_grouped', 'fuel', 'seller_type', 'transmission', 'owner']
    target = 'selling_price'

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Model definitions
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    results = []
    trained_pipelines = {}

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    for name, model in candidate_models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Train Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test Metrics
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results.append({
            'Model': name,
            'Train MAE (₹)': train_mae,
            'Train RMSE (₹)': train_rmse,
            'Train R²': train_r2,
            'Test MAE (₹)': test_mae,
            'Test RMSE (₹)': test_rmse,
            'Test R²': test_r2
        })

        print(f"Test MAE:  ₹{test_mae:,.2f}")
        print(f"Test RMSE: ₹{test_rmse:,.2f}")
        print(f"Test R²:   {test_r2:.4f}")

    results_df = pd.DataFrame(results).sort_values(by='Test R²', ascending=False)
    print("\n=================== MODEL PERFORMANCE COMPARISON ===================")
    print(results_df.to_string(index=False))

    # Identify best performing model
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\nBest Performing Model: {best_model_name}")

    # Save best model pipeline
    best_model_path = os.path.join(models_dir, 'best_car_price_model.joblib')
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved best model artifact to: {best_model_path}")

    # Feature Importance for best model
    if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
        # Extract encoded feature names
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']
        ohe = fitted_preprocessor.named_transformers_['cat']
        encoded_cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        all_feature_names = numeric_features + encoded_cat_names

        importances = best_pipeline.named_steps['regressor'].feature_importances_
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_feats = feat_df.head(15).iloc[::-1]  # Reverse for bottom-up barh

        plt.figure(figsize=(10, 7))
        plt.barh(top_feats['Feature'], top_feats['Importance'], color='#1f77b4', edgecolor='none', height=0.65)
        plt.title(f'Top 15 Feature Importances ({best_model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Relative Importance (Gini / Split Criterion)', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.tight_layout()

        importance_path = os.path.join(figures_dir, 'feature_importance.png')
        plt.savefig(importance_path, dpi=300)
        plt.close()
        print(f"Saved feature importance chart to: {importance_path}")

    # Actual vs. Predicted Plot for Best Model
    y_test_pred = best_pipeline.predict(X_test)
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test / 100000, y_test_pred / 100000, alpha=0.4, color='#2980b9', s=30, label='Predictions')
    max_val = max(y_test.max(), y_test_pred.max()) / 100000
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    plt.title(f'Actual vs. Predicted Selling Price ({best_model_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Actual Price (Lakhs INR)', fontsize=11)
    plt.ylabel('Predicted Price (Lakhs INR)', fontsize=11)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend()
    plt.tight_layout()

    pred_path = os.path.join(figures_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Saved actual vs. predicted plot to: {pred_path}")

    return results_df, best_model_name


if __name__ == "__main__":
    data_file = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/data/processed/cleaned_car_details.csv"
    models_folder = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/models"
    reports_folder = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/reports"

    train_and_evaluate(data_file, models_folder, reports_folder)
