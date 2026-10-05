"""
Map raw Google Forms responses to analyzed long-format data.

This script:
1. Loads the manifest.csv (which maps position 1-100 to true_class and epsilon)
2. Loads the raw wide-format Google Forms CSV
3. Reshapes into a long format: 1 row per participant per image
4. Joins the true_class and epsilon from the manifest
5. Computes response_correct (True if predicted == true_class)
6. Outputs phase3_human_study/data/responses_mapped.csv
7. Prints summary statistics
"""

import csv
import os

# Paths
MANIFEST_FILE = os.path.join(os.path.dirname(__file__), 'manifest.csv')
RAW_DATA_FILE = os.path.join(os.path.dirname(__file__), 'Human Psychophysics Study (Responses) - Form responses 1.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'data', 'responses_mapped.csv')

def load_manifest():
    """Load manifest.csv into a dictionary keyed by integer position."""
    manifest = {}
    with open(MANIFEST_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row['position'])
            manifest[pos] = {
                'image_id': row['image_id'],
                'true_class': row['true_class'],
                'epsilon': row['epsilon']
            }
    return manifest

def main():
    manifest = load_manifest()
    
    long_rows = []
    
    with open(RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Process each participant (each row in the CSV is one participant)
        for p_idx, row in enumerate(reader):
            # Create a unique, anonymous ID for each participant
            # We don't use their name to preserve anonymity in the analysis
            participant_id = f"P{p_idx+1:03d}"
            
            # Extract metadata
            timestamp = row[0]
            vision = row[3]
            device = row[4]
            
            # Only process if they consented
            if "Yes" not in row[1]:
                continue
                
            # Process the 100 image responses
            # The columns are: 
            # 0-4: Metadata
            # 5: Image 1 What
            # 6: Image 1 Conf
            # 7: Image 2 What
            # 8: Image 2 Conf
            # ...
            # 203: Image 100 What
            # 204: Image 100 Conf
            
            for pos in range(1, 101):
                what_col = 5 + 2 * (pos - 1)
                conf_col = 6 + 2 * (pos - 1)
                
                # Check if we have data for this column (some might be blank if they skipped)
                if what_col >= len(row) or not row[what_col].strip():
                    continue
                    
                predicted_class = row[what_col].strip().lower()
                
                # Handle confidence rating (might be empty or non-numeric)
                try:
                    confidence = float(row[conf_col].strip())
                except (ValueError, IndexError):
                    confidence = float('nan')
                    
                # Get truth data from manifest
                truth = manifest[pos]
                true_class = truth['true_class'].lower()
                
                # Compare
                is_correct = (predicted_class == true_class)
                
                long_rows.append({
                    'participant_id': participant_id,
                    'timestamp': timestamp,
                    'vision_status': vision,
                    'device': device,
                    'position': pos,
                    'image_id': truth['image_id'],
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'response_correct': is_correct,
                    'confidence_rating': confidence,
                    'epsilon': truth['epsilon'],
                    'attack_type': 'pgd' # Hardcoded as we know these are PGD
                })

    # Save the reshaped data
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'participant_id', 'timestamp', 'vision_status', 'device', 
            'position', 'image_id', 'true_class', 'predicted_class', 
            'response_correct', 'confidence_rating', 'epsilon', 'attack_type'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)
        
    print(f"Processed {len(long_rows)} total responses.")
    print(f"Saved to {OUTPUT_FILE}\n")
    
    # Calculate statistics per epsilon
    epsilons = sorted(list(set(r['epsilon'] for r in long_rows)))
    
    print("Summary Statistics per Epsilon:")
    print("="*60)
    print(f"{'Epsilon':<10} | {'Accuracy':<15} | {'Mean Confidence':<15} | {'Count'}")
    print("-" * 60)
    
    for eps in epsilons:
        subset = [r for r in long_rows if r['epsilon'] == eps]
        if not subset:
            continue
            
        correct_count = sum(1 for r in subset if r['response_correct'])
        acc = (correct_count / len(subset)) * 100
        
        # Calculate mean confidence, ignoring NaNs
        confs = [r['confidence_rating'] for r in subset if r['confidence_rating'] == r['confidence_rating']]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        
        print(f"{eps:<10} | {acc:>6.2f}%         | {mean_conf:>5.2f} / 10.00    | {len(subset)}")

if __name__ == '__main__':
    main()
