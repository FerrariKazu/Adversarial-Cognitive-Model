import pandas as pd
import numpy as np

CSV_PATH = 'phase5_sdt/results/sdt_results_v4.csv'
df = pd.read_csv(CSV_PATH)

systems = df['system'].unique()
epsilons = sorted(df['epsilon'].unique())

print("SYSTEMS:", list(systems))
print("EPSILONS:", epsilons)

print("\nMean d' values:")
print("=" * 80)
print(f"{'System':<15} | " + " | ".join([f"{eps:.2f}" for eps in epsilons]))
print("-" * 80)
for sys in sorted(systems):
    row_strs = []
    for eps in epsilons:
        sub = df[(df['system'] == sys) & (np.abs(df['epsilon'] - eps) < 1e-4)]
        if len(sub) == 0:
            row_strs.append("NaN")
        else:
            mean_d = sub['d_prime'].mean()
            row_strs.append(f"{mean_d:.3f}")
    print(f"{sys:<15} | " + " | ".join(row_strs))
print("=" * 80)
