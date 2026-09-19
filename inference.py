"""
Sample Inference Script for Used Car Price Prediction.
Loads trained best model pipeline and predicts prices on unseen sample cars.
"""

import os
import joblib
import pandas as pd


def predict_car_price(model_path: str, car_features: pd.DataFrame) -> pd.Series:
    """Predict car price given input feature DataFrame."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    
    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(car_features)
    return predictions


if __name__ == "__main__":
    model_file = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/models/best_car_price_model.joblib"

    # Define test sample vehicles
    samples = pd.DataFrame([
        {
            'brand_grouped': 'Maruti',
            'car_age': 5,
            'km_driven': 45000,
            'fuel': 'Petrol',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner',
            'mileage': 21.4,
            'engine': 1197.0,
            'max_power': 82.0,
            'seats': 5
        },
        {
            'brand_grouped': 'BMW',
            'car_age': 4,
            'km_driven': 30000,
            'fuel': 'Diesel',
            'seller_type': 'Dealer',
            'transmission': 'Automatic',
            'owner': 'First Owner',
            'mileage': 18.5,
            'engine': 1995.0,
            'max_power': 190.0,
            'seats': 5
        },
        {
            'brand_grouped': 'Toyota',
            'car_age': 8,
            'km_driven': 95000,
            'fuel': 'Diesel',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'Second Owner',
            'mileage': 12.8,
            'engine': 2494.0,
            'max_power': 102.0,
            'seats': 7
        }
    ])

    preds = predict_car_price(model_file, samples)
    
    print("=== SAMPLE PREDICTIONS ===")
    for i, row in samples.iterrows():
        print(f"\nCar {i+1}: {row['brand_grouped']} ({row['transmission']}, {row['fuel']}, {row['car_age']} yrs old, {row['km_driven']:,} km, {row['max_power']} bhp)")
        print(f"Predicted Selling Price: ₹{preds[i]:,.2f} ({preds[i]/100000:.2f} Lakhs INR)")
