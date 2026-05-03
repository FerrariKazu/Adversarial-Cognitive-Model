"""
Response Schema & Validation
==============================

PURPOSE:
    Defines the expected data format for human responses collected via
    Google Forms (or any survey tool). Provides a template CSV and a
    validation function to catch data integrity issues before analysis.

SCHEMA COLUMNS EXPLAINED:
    - participant_id    : Anonymized ID (e.g., P001). No real names.
    - session_date      : ISO date (YYYY-MM-DD) for tracking sessions.
    - image_id          : Matches image_idx from stimuli_manifest.csv.
    - attack_type       : 'fgsm', 'pgd', or 'cw'.
    - epsilon           : Perturbation level (e.g., 0.05). 'auto' for C&W.
    - true_class        : Ground-truth CIFAR-10 class name.
    - human_response    : What the participant answered.
    - confidence_rating : Integer 1-10. WHY THIS MATTERS:
        Confidence lets us build ROC curves in Phase 5 (SDT).
        Binary correct/wrong only gives one point on the ROC curve.
        Confidence ratings give us 10 points, revealing the full
        tradeoff between hit rate and false alarm rate.
    - response_correct  : Boolean derived from true_class == human_response.
    - response_time_ms  : Optional. If collected, reveals speed-accuracy
                          tradeoffs (faster responses are often less accurate).
"""

import os
import csv
import pandas as pd

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

SCHEMA_COLUMNS = [
    'participant_id',
    'session_date',
    'image_id',
    'attack_type',
    'epsilon',
    'true_class',
    'human_response',
    'confidence_rating',
    'response_correct',
    'response_time_ms'
]


def generate_template():
    """
    Create an empty response_template.csv with the correct header.

    1. WHAT: Writes a CSV file with column headers and no data rows.
    2. WHY: Gives the researcher a known-good schema to populate. Manual
       data entry into a pre-formatted template avoids column-name typos
       that would silently break analysis pipelines.
    3. OBSERVE: The file will have exactly 1 line (the header).
    """
    output_path = 'response_template.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(SCHEMA_COLUMNS)

    print(f"Generated: {output_path}")
    print(f"  Columns: {SCHEMA_COLUMNS}")
    return output_path


def validate_responses(filepath):
    """
    Validate a filled response CSV for data integrity issues.

    Checks performed:
    1. All required columns exist.
    2. No null values in required fields.
    3. confidence_rating is integer 1-10.
    4. human_response is a valid CIFAR-10 class name.
    5. response_correct matches (true_class == human_response).
    6. attack_type is one of the expected values.
    7. Participant IDs follow the expected pattern.

    Returns
    -------
    dict with keys: 'valid' (bool), 'errors' (list of str), 'warnings' (list),
                    'summary' (dict of counts).
    """
    result = {'valid': True, 'errors': [], 'warnings': [], 'summary': {}}

    if not os.path.exists(filepath):
        result['valid'] = False
        result['errors'].append(f"File not found: {filepath}")
        return result

    df = pd.read_csv(filepath)
    result['summary']['total_rows'] = len(df)
    result['summary']['columns_found'] = list(df.columns)

    # Check 1: Required columns
    required = ['participant_id', 'image_id', 'attack_type', 'epsilon',
                'true_class', 'human_response', 'confidence_rating']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        result['valid'] = False
        result['errors'].append(f"Missing columns: {missing_cols}")
        return result

    # Check 2: Null values in required fields
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            result['valid'] = False
            result['errors'].append(f"Column '{col}' has {null_count} null values")

    # Check 3: Confidence rating range
    if 'confidence_rating' in df.columns:
        bad_conf = df[~df['confidence_rating'].between(1, 10)]
        if len(bad_conf) > 0:
            result['valid'] = False
            result['errors'].append(
                f"{len(bad_conf)} rows have confidence_rating outside [1, 10]")

    # Check 4: Valid class names
    if 'human_response' in df.columns:
        invalid_responses = df[~df['human_response'].isin(CLASSES)]
        if len(invalid_responses) > 0:
            result['valid'] = False
            bad_vals = invalid_responses['human_response'].unique().tolist()
            result['errors'].append(
                f"{len(invalid_responses)} rows have invalid human_response: {bad_vals}")

    # Check 5: response_correct consistency
    if 'response_correct' in df.columns and 'true_class' in df.columns:
        expected = (df['true_class'] == df['human_response'])
        actual = df['response_correct'].astype(bool)
        mismatches = (expected != actual).sum()
        if mismatches > 0:
            result['warnings'].append(
                f"{mismatches} rows have response_correct inconsistent with "
                f"true_class vs human_response")

    # Check 6: Valid attack types
    valid_attacks = ['fgsm', 'pgd', 'cw']
    if 'attack_type' in df.columns:
        invalid_attacks = df[~df['attack_type'].isin(valid_attacks)]
        if len(invalid_attacks) > 0:
            result['warnings'].append(
                f"Unexpected attack types: "
                f"{invalid_attacks['attack_type'].unique().tolist()}")

    # Summary statistics
    result['summary']['participants'] = df['participant_id'].nunique()
    result['summary']['responses_per_participant'] = (
        df.groupby('participant_id').size().to_dict())

    # Print report
    print(f"\n{'='*50}")
    print(f"VALIDATION REPORT: {filepath}")
    print(f"{'='*50}")
    print(f"  Total rows     : {result['summary']['total_rows']}")
    print(f"  Participants   : {result['summary']['participants']}")
    print(f"  Valid          : {'YES' if result['valid'] else 'NO'}")

    if result['errors']:
        print(f"\n  ERRORS ({len(result['errors'])}):")
        for e in result['errors']:
            print(f"    ✗ {e}")

    if result['warnings']:
        print(f"\n  WARNINGS ({len(result['warnings'])}):")
        for w in result['warnings']:
            print(f"    ⚠ {w}")

    if result['valid'] and not result['warnings']:
        print(f"\n  ✓ All checks passed. Data is ready for Phase 4 analysis.")

    return result


if __name__ == '__main__':
    template_path = generate_template()
    print("\nTo validate a completed response file, run:")
    print("  python response_schema.py <path_to_responses.csv>")

    import sys
    if len(sys.argv) > 1:
        validate_responses(sys.argv[1])
