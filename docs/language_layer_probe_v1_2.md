# Language Layer Probe v1.2 (Prereg Draft)

**Status:** DRAFT - Lowest priority. Forward passes only. No training.

## Question
Does TRADES fine-tuning of DINOv2-S at 112 px preserve alignment with a fixed text encoder?

## Comparison
- DINOv2-S base (pretrained) evaluated at 112 px
- Official ff_112 (TRADES fine-tuned on ImageNet-100 at 112 px)

## Features
- Primary feature: **CLS token** (report mean patch separately as secondary)
- Evaluate both CLS token and mean patch features

## Sample size
- n pairs = 5000 (public image-caption subset)
- k for mutual-kNN = 20

## Metrics
- Mutual-kNN overlap (between vision features and text features)
- Linear CKA (between vision features and text features)

## Decision Rule (proposed, flagged)
- If ff_112 mutual-kNN >= 90% of base -> a text side can attach without re-alignment
- If ff_112 mutual-kNN < 70% of base -> needs a fitted orthogonal map
- Intermediate (70-90%) -> inconclusive

## Constraints
- Forward passes only
- Lowest priority
- Must verify licence
- Use fixed text encoder
- Match resolution (DINOv2-S base at 112 px)
