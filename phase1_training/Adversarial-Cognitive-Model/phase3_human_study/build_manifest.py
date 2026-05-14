"""
Parse responseorder.txt → manifest.csv

Maps each image position (1-100) to its true class and epsilon block.
This manifest is the Rosetta Stone for joining Google Forms responses
(which are just numbered column pairs) back to ground truth.
"""

import csv
import re

INPUT_FILE = 'phase3_human_study/responseorder.txt'
OUTPUT_FILE = 'phase3_human_study/manifest.csv'


def parse_response_order(filepath):
    """
    Parse the block-structured responseorder.txt file.

    Format:
        block N - epsilon = X.XX
        <blank>
        classname_imageid
        classname_imageid
        ...
        <blank>

    Returns list of dicts: [{position, image_id, true_class, epsilon}, ...]
    """
    entries = []
    current_epsilon = None
    position = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            # Match block header: "block N - epsilon = X.XX"
            block_match = re.match(r'block\s+\d+\s*-\s*epsilon\s*=\s*([\d.]+)', line, re.IGNORECASE)
            if block_match:
                current_epsilon = float(block_match.group(1))
                continue

            # Must be an image ID like "dog_04224"
            # Extract class name: everything before the last underscore+digits
            img_match = re.match(r'^([a-z]+)_(\d+)$', line)
            if img_match and current_epsilon is not None:
                position += 1
                class_name = img_match.group(1)
                entries.append({
                    'position': position,
                    'image_id': line,
                    'true_class': class_name,
                    'epsilon': f'{current_epsilon:.2f}',
                })

    return entries


def main():
    entries = parse_response_order(INPUT_FILE)

    print(f"Parsed {len(entries)} images from responseorder.txt")
    print()

    # Sanity checks
    assert len(entries) == 100, f"Expected 100 images, got {len(entries)}"

    # Count per block
    from collections import Counter
    eps_counts = Counter(e['epsilon'] for e in entries)
    for eps in sorted(eps_counts):
        print(f"  ε={eps}: {eps_counts[eps]} images")

    # Write CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['position', 'image_id', 'true_class', 'epsilon'])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\n✅ Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
