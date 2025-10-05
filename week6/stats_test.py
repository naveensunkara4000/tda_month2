import pandas as pd
from scipy import stats
import os

data_path = "../data/client_data.csv"
output_dir = "../outputs/week6"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

# T-test: Delhi vs Mumbai temperatures
delhi = df[df['city']=='Delhi']['temperature']
mumbai = df[df['city']=='Mumbai']['temperature']

t_stat, p_val = stats.ttest_ind(delhi, mumbai)
with open(os.path.join(output_dir, "ttest_result.txt"), "w") as f:
    f.write(f"T-statistic: {t_stat}\nP-value: {p_val}\n")

print("Week 6 T-test done! Check outputs/week6/")
