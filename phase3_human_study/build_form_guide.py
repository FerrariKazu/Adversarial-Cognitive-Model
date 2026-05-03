"""
Google Forms Construction Guide Generator
==========================================

PURPOSE:
    Generates a step-by-step text file for building a Google Forms survey
    for the human psychophysics experiment.

DESIGN NOTES:
    - Uses blocked ascending design (easy → hard epsilon levels)
    - Includes informed consent, participant info, and debrief sections
    - 5 blocks × 20 images = 100 total images per participant
    - Each image gets 2 questions: classification + confidence (1-10)
"""

import os

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
EPSILONS = ['0.00', '0.05', '0.10', '0.20', '0.30']
IMAGES_PER_BLOCK = 20


def generate_form_guide():
    guide = []
    guide.append("=" * 70)
    guide.append("GOOGLE FORMS CONSTRUCTION GUIDE")
    guide.append("Adversarial Cognitive Model — Human Psychophysics Study")
    guide.append("=" * 70)

    # CONSENT SECTION
    guide.append("\n--- SECTION 1: INFORMED CONSENT (Page 1) ---")
    guide.append("""
Title: "Informed Consent"
Description (paste exactly):

  You are invited to participate in a research study on visual perception.
  PURPOSE: This study examines how humans classify images that have been
  subtly modified by computer algorithms.
  WHAT YOU WILL DO: View ~100 images and answer two questions about each.
  Takes approximately 20-30 minutes.
  RISKS: None. Images are everyday objects.
  VOLUNTARY: You may stop at any time without penalty.
  PRIVACY: Responses will be anonymized. No identifying info is linked
  to your answers. Data used for academic research only.

Question: "I consent to participate in this study."
Options: "Yes, I consent" / "No, I do not consent"
Required: YES
""")

    # PARTICIPANT INFO
    guide.append("--- SECTION 2: PARTICIPANT INFORMATION (Page 2) ---")
    guide.append("""
Q1 (Short Answer): "Enter your participant ID (e.g., P001):" — REQUIRED
Q2 (Multiple Choice): "Normal or corrected-to-normal vision?" — Yes/No
Q3 (Multiple Choice): "Device type?" — Desktop / Tablet / Smartphone
""")

    # INSTRUCTIONS
    guide.append("--- SECTION 3: INSTRUCTIONS (Page 3) ---")
    guide.append("""
Description: "You will see images containing ONE object from these categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
For each image answer: (1) What object? (2) Confidence 1-10.
Some images may appear noisy. Answer based on first impression.
5 blocks of 20 images each. ~20-30 minutes total."
""")

    # IMAGE BLOCKS
    for i, eps in enumerate(EPSILONS):
        guide.append(f"--- SECTION {4+i}: BLOCK {i+1} — Epsilon={eps} ---")
        guide.append(f"Title: 'Block {i+1} of 5'")
        guide.append(f"Images from: stimuli/pgd/eps{eps}/")
        guide.append(f"Select 2 images per class (20 total).\n")
        guide.append("For EACH image add:")
        guide.append("  1. Insert image (upload the PNG)")
        guide.append("  2. Multiple Choice: 'What object is shown?'")
        guide.append(f"     Options: {', '.join(CLASSES)}")
        guide.append("  3. Linear Scale: 'How confident?' 1(guess) to 10(certain)")
        guide.append("")

    # DEBRIEF
    guide.append("--- FINAL SECTION: DEBRIEF ---")
    guide.append("""
Title: "Thank You"
Description: "Thank you for participating! You were shown images altered
by adversarial perturbation algorithms. Your responses help us compare
human and machine visual perception."
""")

    output_path = 'form_structure.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(guide))

    print(f"Generated: {output_path}")
    print(f"  Blocks: {len(EPSILONS)}, Images/block: {IMAGES_PER_BLOCK}")
    print(f"  Total images: {len(EPSILONS) * IMAGES_PER_BLOCK}")


if __name__ == '__main__':
    generate_form_guide()
