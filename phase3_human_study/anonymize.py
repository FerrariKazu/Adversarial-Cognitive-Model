"""
Response Data Anonymizer
=========================

PURPOSE:
    Takes a raw response CSV (which may contain participant names, emails,
    or other identifying information) and replaces them with anonymous IDs.

WHY ANONYMIZATION IS NECESSARY:
    1. ETHICAL: Participants were promised their data would be anonymized.
       Keeping real names in analysis files violates this promise.
    2. LEGAL: Under GDPR, FERPA, and most university data policies,
       personally identifiable information (PII) in research data must be
       either anonymized or stored under strict security protocols.
    3. SCIENTIFIC: Anonymized data can be shared openly (e.g., on GitHub
       or in supplementary materials), enabling reproducibility.

    The anonymization is ONE-WAY: there is no mapping file saved that
    could re-identify participants. This is by design — once anonymized,
    the data cannot be traced back to individuals.
"""

import os
import sys
import pandas as pd


def anonymize_responses(input_path, output_path=None):
    """
    Replace participant identifiers with anonymous sequential IDs.

    Parameters
    ----------
    input_path  : str — Path to raw response CSV.
    output_path : str — Path for anonymized output. Defaults to
                        'anonymized_responses.csv' in the same directory.

    Returns
    -------
    str — Path to the anonymized file.

    WHAT THIS DOES:
    1. Reads the raw CSV.
    2. Finds all unique values in 'participant_id'.
    3. Maps each to 'participant_001', 'participant_002', etc.
    4. Strips any columns that might contain PII (email, name, IP).
    5. Saves the cleaned CSV.
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(input_path), 'anonymized_responses.csv')

    df = pd.read_csv(input_path)
    print(f"Loaded: {input_path} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # Step 1: Build anonymization mapping
    # -----------------------------------------------------------------
    # 1. WHAT: Create a deterministic mapping from original IDs to anonymous ones.
    # 2. WHY: Using sorted unique values ensures the mapping is deterministic
    #         (same input always produces same output), which is important for
    #         debugging but does NOT compromise anonymity.
    # 3. OBSERVE: The mapping is NOT saved to disk — it exists only in memory.
    # -----------------------------------------------------------------
    if 'participant_id' in df.columns:
        unique_ids = sorted(df['participant_id'].unique())
        id_map = {
            original: f"participant_{i+1:03d}"
            for i, original in enumerate(unique_ids)
        }
        df['participant_id'] = df['participant_id'].map(id_map)
        print(f"  Anonymized {len(unique_ids)} participant IDs")

    # -----------------------------------------------------------------
    # Step 2: Strip potential PII columns
    # -----------------------------------------------------------------
    # 1. WHAT: Remove columns that might contain personally identifiable info.
    # 2. WHY: Google Forms sometimes includes a "Timestamp" column with exact
    #         submission times, or an email column if "Collect email addresses"
    #         was accidentally left on. These could de-anonymize participants.
    # 3. OBSERVE: Only removes columns if they exist — safe on clean data.
    # -----------------------------------------------------------------
    pii_columns = ['email', 'name', 'full_name', 'ip_address',
                   'Email Address', 'Email', 'Name', 'Timestamp']
    dropped = [col for col in pii_columns if col in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"  Removed PII columns: {dropped}")

    # -----------------------------------------------------------------
    # Step 3: Save anonymized data
    # -----------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"  ✓ Anonymization complete. No re-identification mapping was saved.")

    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python anonymize.py <raw_responses.csv> [output_path.csv]")
        print("\nThis script anonymizes participant data for research compliance.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    anonymize_responses(input_file, output_file)
