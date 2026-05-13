import os
import pandas as pd
import re
import csv

def main():
    base_dir = os.path.dirname(__file__)
    response_order_file = os.path.join(base_dir, 'responseorder.txt')
    responses_file = os.path.join(base_dir, 'Human Psychophysics Study (Responses) - Form responses 1.csv')
    output_dir = os.path.join(base_dir, 'data')
    output_file = os.path.join(output_dir, 'responses_mapped.csv')

    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse responseorder.txt
    image_order = []
    current_epsilon = None
    with open(response_order_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('block'):
                # Extract epsilon
                match = re.search(r'epsilon = ([0-9.]+)', line)
                if match:
                    current_epsilon = float(match.group(1))
            else:
                image_id = line
                true_class = image_id.split('_')[0]
                image_order.append({
                    'image_id': image_id,
                    'epsilon': current_epsilon,
                    'true_class': true_class,
                    'attack_type': 'pgd'  # Assuming PGD for the study
                })

    print(f"Parsed {len(image_order)} images from responseorder.txt")

    # 2. Read the responses CSV
    df = pd.read_csv(responses_file)
    print(f"Read {len(df)} responses from CSV")

    # Columns 0-4 are metadata: Timestamp, Consent, Name, Vision, Device
    # Columns 5 to 204 are the image responses: "What object is shown?", "How confident?"
    
    # Check if the number of image columns matches the expected 100 * 2 = 200
    expected_image_cols = len(image_order) * 2
    actual_image_cols = len(df.columns) - 5
    print(f"Expected image columns: {expected_image_cols}, Actual: {actual_image_cols}")
    
    if expected_image_cols != actual_image_cols:
        print("Warning: Column count mismatch!")

    # 3. Transform to long format
    long_records = []
    for index, row in df.iterrows():
        # Generate a participant ID
        participant_name = row.iloc[2]
        if pd.isna(participant_name) or participant_name.strip() == '' or participant_name == '.':
            participant_id = f"P{index+1:03d}"
        else:
            # We can still use PXXX to keep it semi-anonymous in the output, or just use the name hash
            participant_id = f"P{index+1:03d}"

        session_date = row.iloc[0].split(' ')[0] # "06/05/2026 00:38:42" -> "06/05/2026"

        for i, img_info in enumerate(image_order):
            col_idx_obj = 5 + (i * 2)
            col_idx_conf = 6 + (i * 2)
            
            human_response = str(row.iloc[col_idx_obj]).lower().strip()
            # If the user selected 'automobile' but the dataset uses 'automobile', ensure it matches.
            # CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
            
            try:
                confidence_rating = int(row.iloc[col_idx_conf])
            except (ValueError, TypeError):
                confidence_rating = None

            response_correct = (human_response == img_info['true_class'])

            record = {
                'participant_id': participant_id,
                'session_date': session_date,
                'image_id': img_info['image_id'],
                'attack_type': img_info['attack_type'],
                'epsilon': img_info['epsilon'],
                'true_class': img_info['true_class'],
                'human_response': human_response,
                'confidence_rating': confidence_rating,
                'response_correct': response_correct,
                'response_time_ms': '' # Not collected
            }
            long_records.append(record)

    df_long = pd.DataFrame(long_records)
    
    # 4. Save to CSV
    # Ensure columns match the SCHEMA_COLUMNS from response_schema.py
    schema_columns = [
        'participant_id', 'session_date', 'image_id', 'attack_type',
        'epsilon', 'true_class', 'human_response', 'confidence_rating',
        'response_correct', 'response_time_ms'
    ]
    df_long = df_long[schema_columns]
    df_long.to_csv(output_file, index=False)
    print(f"Saved {len(df_long)} records to {output_file}")

if __name__ == '__main__':
    main()
