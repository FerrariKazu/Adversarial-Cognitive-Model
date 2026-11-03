# Language Layer Probe v1 (Prereg Draft)

**Status:** DRAFT - Lowest priority. Forward passes only. No training.

## Question
Does TRADES fine-tuning of DINOv2-S at 112 px preserve alignment with a fixed text encoder?

## Comparison
- DINOv2-S base (pretrained)
- Official ff_112 (TRADES fine-tuned on ImageNet-100 at 112 px)

## Metrics
- Mutual-kNN overlap (between vision features and text features)
- Linear CKA (between vision features and text features)

## Data
- Public image-caption set (licence must be checked before use)
- Candidates to evaluate for licence: COCO captions (check terms), Flickr30k (check terms), Conceptual Captions subsets (check terms). Only use if licence is compatible.

## Text Encoder
- Fixed off-the-shelf encoder: prefer `all-mpnet-base-v2` (sentence-transformers) or CLIP text encoder as fallback.
- Check if authors' text JEPA is released; if yes, consider using it instead, but default to off-the-shelf fixed encoder.

## Procedure
1. Extract vision features from both models (DINOv2-S base, ff_112) on the image side of public image-caption pairs
2. Extract text features from fixed text encoder on captions
3. Compute mutual-kNN overlap and linear CKA
4. Report results; no fine-tuning of text encoder, forward passes only

## Constraints
- Forward passes only (no training)
- Lowest priority
- Must verify licence of image-caption dataset before use
- Use fixed text encoder (do not train)
