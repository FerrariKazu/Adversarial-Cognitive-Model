# Data Feasibility (Draft)

**Status:** DRAFT - Download nothing without approval. All entries are candidates requiring verification of licence/access.

## Direction 1: Budgeted search for small rare findings in huge 2D images
| Dataset | Size | Access/Licence | Split policy (patient/lesion-grouped) | Label availability | Fits 8GB card? | Notes |
|---|---|---|---|---|---|---|
| CAMELYON16/17 (WSI metastases) | Large WSIs (varies) | Often requires registration; licence varies by challenge | Patient/slide-level grouping critical | Metastasis labels (slide/patch level) | Patch-level feasible if tiled; full WSIs large - need careful caching | Small rare findings in large images; patching required |
| KiTS/other lesion detection | CT volumes/slices, moderate | Public with licence terms (check) | Patient-level split required | Lesion annotations | 2D slices feasible; volumes may need subsampling | 2D slice-based possible |
| ISIC (skin lesions) | ~10k+ images | Public with terms (check specific version) | Often image-level; patient grouping may be incomplete | Lesion labels/classification | Yes, easily fits 8GB | Smaller, less "huge image" scale |
| RSNA Pneumonia/Mammography challenges | Varies | Challenge terms apply | Patient-level grouping required | Localized findings | Feasible with care | Check current licence terms |

## Direction 2: Active frame/slice selection in cine and 4D
| Dataset | Size | Access/Licence | Split policy (patient/lesion-grouped) | Label availability | Fits 8GB card? | Notes |
|---|---|---|---|---|---|---|
| ACDC (Automated Cardiac Diagnosis Challenge) | Cine MRI, ~150 subjects | Public (challenge licence) - verify current terms | Patient-level split required | Cardiac segmentation/diagnosis labels | Yes, fits 8GB (small) | Public cine set; good for prototyping |
| M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge) | Cine MRI, multi-site ~375 subjects | Public challenge data with terms - verify | Patient/site-level grouping critical for shift | Segmentation/labels | Yes, fits 8GB | Multi-site makes it useful for shift testing |
| CMRxRecon (Cardiac MRI Reconstruction Challenge) | k-space + images, multi-coil cine | Public challenge data with terms - verify | Patient-level split required | Reconstruction targets; some segmentation available | Depends on subsampling; raw k-space large but image-domain subsets feasible | Public cine/recon set; verify access terms |
| UK Biobank cardiac imaging (subset) | Very large | Controlled access, not openly public | Patient-level, strict governance | Rich labels | Unlikely to fit simple download without approval; access restricted | Not freely downloadable |

## Direction 3: Robustness and shift testing of active-perception where/stop decisions
| Dataset | Size | Access/Licence | Split policy (patient/lesion-grouped) | Label availability | Fits 8GB card? | Notes |
|---|---|---|---|---|---|---|
| M&Ms | Multi-site cine MRI | Public challenge (verify terms) | Site/patient grouped splits essential for shift | Segmentation/diagnosis | Yes | Good for scanner/site shift |
| Camelyon (domain shift) | WSI across centers | Public with terms (verify) | Center/hospital grouped splits | Metastasis labels | Patch-level feasible | Histopathology shift |
| CheXpert/CXR (if considering 2D) | Large but public subsets exist | Check licence terms | Patient-level split; often used | Labels | Subsets feasible | May need careful licence review |
| ACDC (single site) | Small | Public | Patient-level | Labels | Yes | Less shift, good for calibration baselines |

**General notes:**
- All datasets require licence verification before use. Patient/lesion-grouped splits are mandatory to avoid leakage.
- 8GB card constraint means working with subsets, patches, or preprocessed small volumes; full large datasets likely need streaming/subsampling.
- No downloads planned without explicit approval.
- Public cine sets identified: ACDC, M&Ms, CMRxRecon (all require verifying current access/licence terms).
