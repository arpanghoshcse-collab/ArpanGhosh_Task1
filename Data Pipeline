"""
Data Preprocessing and Feature Engineering Pipeline
for Used Car Price Prediction.
"""

import os
import re
import numpy as np
import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded raw dataset with shape: {df.shape}")
    return df


def clean_and_preprocess(df: pd.DataFrame, reference_year: int = 2024) -> pd.DataFrame:
    """
    Clean and engineer features from the raw CarDekho dataset:
      - Removes duplicate entries
      - Drops rows with missing core technical specs
      - Extracts numeric values from string columns: mileage, engine, max_power
      - Handles corrupted entries (e.g. ' bhp', 0.0 mileage)
      - Standardizes inconsistent categorical values (e.g. casing, whitespace)
      - Feature engineering: car_age (from year) and brand (from name)
    """
    data = df.copy()
    initial_rows = len(data)

    # 1. Deduplication
    data = data.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(data)
    print(f"Removed {initial_rows - dedup_rows} duplicate rows. Remaining: {dedup_rows}")

    # 2. Extract numeric values from string specs using regex
    # mileage: e.g. '23.4 kmpl' or '17.3 km/kg' -> 23.4
    data['mileage_clean'] = data['mileage'].astype(str).str.extract(r'([\d.]+)')[0].astype(float)

    # engine: e.g. '1248 CC' -> 1248.0
    data['engine_clean'] = data['engine'].astype(str).str.extract(r'([\d.]+)')[0].astype(float)

    # max_power: e.g. '74 bhp', corrupted entries like ' bhp' or 'null' become NaN
    data['max_power_clean'] = pd.to_numeric(
        data['max_power'].astype(str).str.extract(r'([\d.]+)')[0],
        errors='coerce'
    )

    # Replace 0.0 mileage with NaN (erroneous test/missing readings)
    data.loc[data['mileage_clean'] <= 0, 'mileage_clean'] = np.nan

    # 3. Handle missing values
    # Drop records that lack essential mechanical specifications
    clean_cols = ['mileage_clean', 'engine_clean', 'max_power_clean', 'seats']
    data = data.dropna(subset=clean_cols).reset_index(drop=True)
    print(f"Filtered out incomplete/corrupted records. Clean records count: {len(data)}")

    # Assign cleaned numeric columns
    data['mileage'] = data['mileage_clean']
    data['engine'] = data['engine_clean']
    data['max_power'] = data['max_power_clean']
    data['seats'] = data['seats'].astype(int)
    data = data.drop(columns=['mileage_clean', 'engine_clean', 'max_power_clean'])

    # 4. Standardize categorical variables (address inconsistent casing, trailing whitespace)
    data['fuel'] = data['fuel'].astype(str).str.strip().str.capitalize()
    data['seller_type'] = data['seller_type'].astype(str).str.strip().str.title()
    data['transmission'] = data['transmission'].astype(str).str.strip().str.capitalize()
    data['owner'] = data['owner'].astype(str).str.strip().str.title()

    # 5. Feature Engineering: Extract brand from car name
    # First token is the brand name
    raw_brand = data['name'].astype(str).str.split().str[0].str.strip().str.title()
    
    # Handle composite brand names like 'Land Rover'
    is_land_rover = (raw_brand == 'Land') & (data['name'].astype(str).str.split().str[1].str.title() == 'Rover')
    raw_brand = np.where(is_land_rover, 'Land Rover', raw_brand)
    
    # Standardize acronyms
    brand_replacements = {
        'Bmw': 'BMW',
        'Mg': 'MG',
        'Mercedes-Benz': 'Mercedes-Benz'
    }
    data['brand'] = raw_brand
    data['brand'] = data['brand'].replace(brand_replacements)

    # Group rare brands (< 10 cars) into 'Other' to avoid sparse one-hot explosions
    brand_counts = data['brand'].value_counts()
    frequent_brands = brand_counts[brand_counts >= 10].index
    data['brand_grouped'] = data['brand'].apply(lambda b: b if b in frequent_brands else 'Other')

    # 6. Feature Engineering: Calculate car age
    # Age calculated relative to reference year (or data collection reference)
    data['car_age'] = reference_year - data['year']
    data['car_age'] = data['car_age'].clip(lower=0)

    print("Data cleaning & feature engineering complete.")
    print(f"Final clean dataset shape: {data.shape}")
    return data


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed dataset to: {filepath}")


if __name__ == "__main__":
    raw_path = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/data/raw/car_details_v3.csv"
    processed_path = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/data/processed/cleaned_car_details.csv"
    
    raw_df = load_raw_data(raw_path)
    clean_df = clean_and_preprocess(raw_df)
    save_processed_data(clean_df, processed_path)
