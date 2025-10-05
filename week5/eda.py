import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
data_path = "../data/client_data.csv"
output_dir = "../outputs/week5"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)

# Summary statistics
summary = df.describe(include='all').T
summary.to_csv(os.path.join(output_dir, "summary_stats.csv"))

# Missing values
missing = df.isnull().sum()
missing.to_csv(os.path.join(output_dir, "missing_values.csv"))

# Correlation heatmap
num_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(6,5))
sns.heatmap(num_cols.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Histograms
for col in num_cols.columns:
    plt.figure()
    num_cols[col].hist(bins=10)
    plt.title(f"Histogram of {col}")
    plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
    plt.close()

print("Week 5 EDA done! Check outputs/week5/")
