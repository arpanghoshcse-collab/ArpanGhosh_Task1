"""
Exploratory Data Analysis (EDA) module for Car Price Prediction.
Generates comprehensive visualization figures and saves them to reports/figures/.
"""

import os
import numpy as np
import pandas as pd

os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Styling settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_all_eda_plots(data_path: str, output_dir: str) -> None:
    """Generate all required EDA visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Loaded dataset for EDA: {df.shape}")

    # 1. Distribution of Selling Prices (Normal and Log-transformed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw price distribution
    sns.histplot(df['selling_price'] / 100000, kde=True, ax=axes[0], color='#2b5c8f', bins=35)
    axes[0].set_title('Distribution of Selling Prices (Lakh INR)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Selling Price (in Lakhs INR)')
    axes[0].set_ylabel('Frequency')
    
    # Log-transformed price distribution
    sns.histplot(np.log1p(df['selling_price']), kde=True, ax=axes[1], color='#e67e22', bins=35)
    axes[1].set_title('Log-Transformed Selling Price Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('log(1 + Selling Price)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'selling_price_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Price vs. Fuel Type Box Plots
    plt.figure(figsize=(9, 6))
    order_fuel = df.groupby('fuel')['selling_price'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x='fuel',
        y=df['selling_price'] / 100000,
        hue='fuel',
        legend=False,
        order=order_fuel,
        palette='Blues_r',
        showfliers=False
    )
    plt.title('Selling Price vs. Fuel Type (Excl. Outliers for Clarity)', fontsize=14, fontweight='bold')
    plt.xlabel('Fuel Type', fontsize=11)
    plt.ylabel('Selling Price (Lakhs INR)', fontsize=11)
    
    fuel_path = os.path.join(output_dir, 'price_vs_fuel_boxplot.png')
    plt.tight_layout()
    plt.savefig(fuel_path, dpi=300)
    plt.close()
    print(f"Saved: {fuel_path}")

    # 3. Price vs. Car Age Scatter Plot (with Trend Line)
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x='car_age',
        y=df['selling_price'] / 100000,
        scatter_kws={'alpha': 0.35, 'color': '#16a085', 's': 25},
        line_kws={'color': '#c0392b', 'linewidth': 2.5}
    )
    plt.title('Selling Price vs. Car Age (Depreciation Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('Car Age (Years)', fontsize=11)
    plt.ylabel('Selling Price (Lakhs INR)', fontsize=11)
    plt.ylim(0, 40)  # Focus on bulk of the market (< 40 Lakhs)
    
    age_path = os.path.join(output_dir, 'price_vs_car_age_scatter.png')
    plt.tight_layout()
    plt.savefig(age_path, dpi=300)
    plt.close()
    print(f"Saved: {age_path}")

    # 4. Feature Correlation Heatmap
    num_cols = ['selling_price', 'car_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    corr_matrix = df[num_cols].corr()

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
    
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")

    # 5. Additional: Average Selling Price by Top 12 Brands
    top_brands = df['brand'].value_counts().head(12).index
    brand_price = df[df['brand'].isin(top_brands)].groupby('brand')['selling_price'].mean().sort_values(ascending=False) / 100000
    
    plt.figure(figsize=(11, 5))
    sns.barplot(x=brand_price.index, y=brand_price.values, hue=brand_price.index, legend=False, palette='viridis')
    plt.title('Average Selling Price for Top 12 Most Common Brands', fontsize=14, fontweight='bold')
    plt.xlabel('Brand', fontsize=11)
    plt.ylabel('Mean Selling Price (Lakhs INR)', fontsize=11)
    plt.xticks(rotation=45)
    
    brand_path = os.path.join(output_dir, 'top_brands_price_barplot.png')
    plt.tight_layout()
    plt.savefig(brand_path, dpi=300)
    plt.close()
    print(f"Saved: {brand_path}")

    print("All EDA visualizations successfully generated and saved!")


if __name__ == "__main__":
    processed_csv = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/data/processed/cleaned_car_details.csv"
    figures_dir = "/Users/computerfive/.gemini/antigravity/scratch/car_price_prediction/reports/figures"
    generate_all_eda_plots(processed_csv, figures_dir)
