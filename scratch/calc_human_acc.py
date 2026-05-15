import csv
import os

# 1. Parse true labels and epsilons
true_labels = []
epsilons = []
current_eps = None

with open('phase3_human_study/responseorder.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if line.startswith('block'):
            # e.g., "block 1 - epsilon = 0.00"
            parts = line.split('=')
            current_eps = float(parts[-1].strip())
        else:
            # e.g., "dog_04224"
            label = line.split('_')[0].strip().lower()
            true_labels.append(label)
            epsilons.append(current_eps)

# 2. Parse human responses
correct_counts = {eps: 0 for eps in set(epsilons)}
total_counts = {eps: 0 for eps in set(epsilons)}

with open('phase3_human_study/human_responses_reconstructed_wide.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    for row in reader:
        if not row[0]: continue  # skip empty
        
        # Responses start at index 5 and alternate every 2 columns
        for i in range(100):
            col_idx = 5 + (i * 2)
            if col_idx < len(row):
                prediction = row[col_idx].strip().lower()
                true_label = true_labels[i]
                eps = epsilons[i]
                
                if prediction == true_label:
                    correct_counts[eps] += 1
                total_counts[eps] += 1

# 3. Print results
print("\n" + "=" * 35)
print("HUMAN PGD ACCURACY")
print("=" * 35)
print(f"{'Epsilon':<10} | {'Human Acc':<12}")
print("-" * 35)
for eps in sorted(set(epsilons)):
    acc = (correct_counts[eps] / total_counts[eps]) * 100 if total_counts[eps] > 0 else 0
    print(f"{eps:<10.2f} | {acc:<5.2f}%")

