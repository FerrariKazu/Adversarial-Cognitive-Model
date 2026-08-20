# Stage 2 Run Log — HPC-only (matrix C): 60-epoch training + three-way Step C eval

> Recorded 2026-08-16 from the Kaggle run that completed Step B (60/60 epochs) and hit the 12 h timeout mid-Step-C PGD-100. Updated 2026-08-18 with the **completed three-way PGD-50 eval** (A baseline / B AIS-v1 / C HPC-only) from the re-run under commit `3c50d51` (per-cell `--resume` + per-leg HF sync). Updated 2026-08-20 with the **definitive 8-seed reruns** (`datasets==4.7.0` pinned): AIS-v1 vs baseline (§8, +7.75 pp NOT significant) and HPC-only vs baseline (§9, +3.92/+4.29 pp NOT significant) — both confirm the pre-registered Stage 1 verdict. Per-epoch numbers below are the exact rows of the trainer diag `report/rhan_next_hpc_only_diag.jsonl` (synced to HF); the printed diagnostic blocks in the run log are those same rows. Step C numbers are the exact HF-synced sweep CSVs (`report/sweep_stage2_hpc_only/epsilon_sweep_*.csv`).

## 0. Run metadata

| field | value |
|---|---|
| config | `RHANNextConfig([HPC(L=1)])` — enable_ais=False, enable_hpc=True, hpc_num_levels=1, w_hpc=0.10 |
| params | 76,663,734 |
| base checkpoint | `checkpoints/rhan_next_ais_v1_halting_only_best.pth` (validated AIS-v1) |
| resume | HF rolling `rhan_next_hpc_only_rolling.pth` @ epoch 34 → resumed from epoch 35 (best val 56.81% at resume) |
| curriculum | 1–20 @ ε=0.031 (prior session) → 21–40 @ ε=0.062 (lr 0.002) → 41–60 @ ε=0.094 (lr 0.001) |
| dataset | 5000 real + 41656 pseudo + 0 synthetic = 46656 (41.7% pseudo-kept) |
| dataloader | num_workers=4, persistent_workers=True, prefetch_factor=4 |
| diag rows | 27 (prepended epoch-1 resume baseline + epochs 35–60) |
| final | Training complete, Best **56.81%**; rolling epoch 60; truck-rank WATCH series logged (27 epochs, final rank=3, margin −0.0169) |
| Step C PGD-50 | **COMPLETE** (three-way A/B/C, 5 seeds × 300 samples, eps ∈ {0.0, 0.094}, norm-space) — 2026-08-18, commit `3c50d51`, provenance `report/sweep_stage2_hpc_only/eval_provenance.json` — **superseded by 8-seed reruns** (§8: AIS-v1 +7.75 pp NOT sig; §9: HPC-only +3.92 pp NOT sig) |
| Step C PGD-100 | **COMPLETE** (three-way A/B/C, 5 seeds × 300 samples, eps=0.094, norm-space) — 2026-08-18, commit `3c50d51`, provenance `report/sweep_stage2_hpc_only_pgd100/eval_provenance.json` — **superseded by 8-seed reruns** (§8: AIS-v1 +7.75 pp NOT sig; §9: HPC-only +4.29 pp NOT sig) |

## 1. Pseudo-label distribution (train-split step, this session)

Total pseudo-labeled: **41656 / 100000 (41.7%)**; combined 5000 real + 41656 pseudo = 46656.

| class | images | mean confidence |
|---|---|---|
| airplane | 6190 | 0.8283 |
| bird | 4545 | 0.7657 |
| car | 5235 | 0.8344 |
| cat | 674 | 0.6975 |
| deer | 3005 | 0.7455 |
| dog | 3520 | 0.7314 |
| horse | 4336 | 0.7647 |
| monkey | 2444 | 0.7056 |
| ship | 5065 | 0.8168 |
| truck | 6642 | 0.8171 |

## 2. Per-epoch diagnostics (epochs 35–60; epoch 1 = prepended resume baseline)

`eps` = adversarial-noise curriculum phase; β_dyn = dynamic β (mean/std); `gate α` = foveal gate; `HPC err` = mean HPC prediction error; `map max/std` = HPC error-map max/std; `truck` = truck Π_D; `r/m` = truck Π_D rank among top-3 / truck−#2 margin. Loss/throughput from the printed epoch lines (not in the diag).

| ep | eps | loss | TrAcc% | TeAcc% | img/s | β_dyn | recon | gate α | HPC err | map max | map std | truck | r | m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.031 | — | 66.5 | 54.6 | — | 1.6403/0.3523 | 1.0507 | 0.4988 | 0.6893 | 1.1578 | 0.2772 | 0.3345 | 3 | -0.0358 |
| 35 | 0.062 | 0.820 | 69.3 | 54.5 | 7.64 | 1.5758/0.3224 | 0.9141 | 0.4966 | 0.1506 | 1.8607 | 0.2662 | 0.2902 | 5 | -0.0315 |
| 36 | 0.062 | 0.819 | 69.4 | 56.1 | 7.68 | 1.5778/0.3248 | 0.9150 | 0.4966 | 0.1510 | 1.8633 | 0.2666 | 0.2975 | 4 | -0.0272 |
| 37 | 0.062 | 0.810 | 69.3 | 56.5 | 7.65 | 1.5762/0.3227 | 0.9110 | 0.4968 | 0.1511 | 1.8641 | 0.2667 | 0.3017 | 3 | -0.0154 |
| 38 | 0.062 | 0.815 | 69.7 | 55.8 | 7.65 | 1.5681/0.3173 | 0.9020 | 0.4966 | 0.1510 | 1.8642 | 0.2666 | 0.2957 | 4 | -0.0168 |
| 39 | 0.062 | 0.814 | 69.6 | 54.8 | 7.66 | 1.5685/0.3157 | 0.8990 | 0.4966 | 0.1508 | 1.8642 | 0.2665 | 0.2987 | 3 | -0.0092 |
| 40 | 0.062 | 0.815 | 70.2 | 54.9 | 7.67 | 1.5724/0.3200 | 0.9060 | 0.4966 | 0.1508 | 1.8646 | 0.2665 | 0.3035 | 3 | -0.0142 |
| 41 | 0.094 | 1.011 | 68.3 | 53.6 | 7.67 | 1.9629/0.3964 | 0.9010 | 0.4968 | 0.1507 | 1.8648 | 0.2665 | 0.2969 | 3 | -0.0203 |
| 42 | 0.094 | 0.970 | 68.0 | 53.8 | 7.66 | 1.9617/0.4006 | 0.8981 | 0.4971 | 0.1506 | 1.8645 | 0.2664 | 0.2971 | 3 | -0.0206 |
| 43 | 0.094 | 0.945 | 67.4 | 54.5 | 7.67 | 1.9637/0.4054 | 0.8940 | 0.4973 | 0.1505 | 1.8646 | 0.2663 | 0.3013 | 3 | -0.0144 |
| 44 | 0.094 | 0.940 | 67.1 | 54.1 | 7.66 | 1.9573/0.3926 | 0.8860 | 0.4978 | 0.1504 | 1.8646 | 0.2663 | 0.2872 | 5 | -0.0254 |
| 45 | 0.094 | 0.932 | 67.2 | 54.1 | 7.65 | 1.9550/0.3907 | 0.8811 | 0.4980 | 0.1503 | 1.8648 | 0.2662 | 0.2955 | 3 | -0.0168 |
| 46 | 0.094 | 0.934 | 67.6 | 52.4 | 7.65 | 1.9538/0.3886 | 0.8777 | 0.4985 | 0.1502 | 1.8647 | 0.2661 | 0.2980 | 3 | -0.0074 |
| 47 | 0.094 | 0.932 | 67.4 | 54.0 | 7.66 | 1.9532/0.3931 | 0.8771 | 0.4985 | 0.1501 | 1.8648 | 0.2661 | 0.2874 | 5 | -0.0236 |
| 48 | 0.094 | 0.936 | 67.8 | 54.5 | 7.66 | 1.9439/0.3815 | 0.8686 | 0.4990 | 0.1500 | 1.8649 | 0.2661 | 0.2930 | 4 | -0.0066 |
| 49 | 0.094 | 0.933 | 67.2 | 53.7 | 7.66 | 1.9524/0.3886 | 0.8766 | 0.4988 | 0.1499 | 1.8651 | 0.2661 | 0.2909 | 4 | -0.0170 |
| 50 | 0.094 | 0.919 | 67.7 | 55.0 | 7.66 | 1.9354/0.3717 | 0.8573 | 0.4988 | 0.1498 | 1.8653 | 0.2661 | 0.2897 | 3 | -0.0168 |
| 51 | 0.094 | 0.912 | 68.3 | 54.6 | 7.65 | 1.9483/0.3940 | 0.8668 | 0.4988 | 0.1497 | 1.8654 | 0.2661 | 0.2875 | 4 | -0.0279 |
| 52 | 0.094 | 0.923 | 67.8 | 55.8 | 7.64 | 1.9453/0.3848 | 0.8634 | 0.4990 | 0.1496 | 1.8658 | 0.2661 | 0.2935 | 3 | -0.0159 |
| 53 | 0.094 | 0.920 | 68.3 | 54.9 | 7.64 | 1.9483/0.3919 | 0.8658 | 0.4990 | 0.1496 | 1.8659 | 0.2661 | 0.2808 | 5 | -0.0294 |
| 54 | 0.094 | 0.922 | 68.3 | 55.4 | 7.63 | 1.9405/0.3818 | 0.8580 | 0.4990 | 0.1495 | 1.8659 | 0.2661 | 0.2860 | 4 | -0.0204 |
| 55 | 0.094 | 0.911 | 68.1 | 55.0 | 7.61 | 1.9404/0.3814 | 0.8538 | 0.4993 | 0.1494 | 1.8661 | 0.2661 | 0.2895 | 3 | -0.0175 |
| 56 | 0.094 | 0.918 | 67.3 | 54.6 | 7.63 | 1.9433/0.3829 | 0.8608 | 0.4993 | 0.1493 | 1.8661 | 0.2661 | 0.2925 | 3 | -0.0148 |
| 57 | 0.094 | 0.905 | 68.6 | 54.8 | 7.62 | 1.9455/0.3884 | 0.8604 | 0.4993 | 0.1493 | 1.8661 | 0.2661 | 0.2944 | 3 | -0.0197 |
| 58 | 0.094 | 0.909 | 68.3 | 55.2 | 7.61 | 1.9462/0.3854 | 0.8648 | 0.4993 | 0.1492 | 1.8663 | 0.2661 | 0.2907 | 4 | -0.0204 |
| 59 | 0.094 | 0.905 | 68.7 | 55.1 | 7.61 | 1.9421/0.3847 | 0.8605 | 0.4993 | 0.1492 | 1.8665 | 0.2661 | 0.2848 | 5 | -0.0188 |
| 60 | 0.094 | 0.916 | 68.8 | 55.1 | 7.64 | 1.9388/0.3804 | 0.8579 | 0.4993 | 0.1492 | 1.8667 | 0.2661 | 0.2896 | 3 | -0.0169 |

## 3. Π_D per class (mean over batch, per epoch)

| ep | airplane | bird | car | cat | deer | dog | horse | monkey | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.3703 | 0.3318 | 0.3718 | 0.2891 | 0.2521 | 0.3333 | 0.2830 | 0.2752 | 0.3093 | 0.3345 |
| 35 | 0.3362 | 0.3014 | 0.3217 | 0.2683 | 0.2324 | 0.2992 | 0.2636 | 0.2480 | 0.2803 | 0.2902 |
| 36 | 0.3247 | 0.2912 | 0.3261 | 0.2711 | 0.2335 | 0.2994 | 0.2673 | 0.2616 | 0.2790 | 0.2975 |
| 37 | 0.3171 | 0.2987 | 0.3219 | 0.2662 | 0.2325 | 0.3000 | 0.2608 | 0.2623 | 0.2814 | 0.3017 |
| 38 | 0.3343 | 0.2939 | 0.3125 | 0.2615 | 0.2286 | 0.2999 | 0.2521 | 0.2509 | 0.2703 | 0.2957 |
| 39 | 0.3216 | 0.2808 | 0.3079 | 0.2713 | 0.2384 | 0.2970 | 0.2659 | 0.2491 | 0.2771 | 0.2987 |
| 40 | 0.3235 | 0.2886 | 0.3177 | 0.2672 | 0.2350 | 0.3030 | 0.2624 | 0.2581 | 0.2679 | 0.3035 |
| 41 | 0.3270 | 0.2884 | 0.3172 | 0.2582 | 0.2345 | 0.2945 | 0.2660 | 0.2567 | 0.2726 | 0.2969 |
| 42 | 0.3216 | 0.2912 | 0.3177 | 0.2699 | 0.2313 | 0.2891 | 0.2617 | 0.2572 | 0.2714 | 0.2971 |
| 43 | 0.3233 | 0.2931 | 0.3157 | 0.2583 | 0.2307 | 0.2984 | 0.2601 | 0.2571 | 0.2757 | 0.3013 |
| 44 | 0.3126 | 0.2831 | 0.3191 | 0.2877 | 0.2291 | 0.2960 | 0.2646 | 0.2535 | 0.2761 | 0.2872 |
| 45 | 0.3226 | 0.2887 | 0.3123 | 0.2626 | 0.2288 | 0.2887 | 0.2617 | 0.2495 | 0.2720 | 0.2955 |
| 46 | 0.3262 | 0.2825 | 0.3054 | 0.2627 | 0.2267 | 0.2904 | 0.2576 | 0.2558 | 0.2762 | 0.2980 |
| 47 | 0.3252 | 0.2981 | 0.3110 | 0.2510 | 0.2266 | 0.2880 | 0.2553 | 0.2566 | 0.2673 | 0.2874 |
| 48 | 0.3074 | 0.2852 | 0.2996 | 0.2530 | 0.2324 | 0.2953 | 0.2571 | 0.2534 | 0.2696 | 0.2930 |
| 49 | 0.3138 | 0.2860 | 0.3079 | 0.2681 | 0.2345 | 0.2938 | 0.2609 | 0.2472 | 0.2749 | 0.2909 |
| 50 | 0.3077 | 0.2743 | 0.3065 | 0.2610 | 0.2258 | 0.2884 | 0.2521 | 0.2443 | 0.2615 | 0.2897 |
| 51 | 0.3179 | 0.2833 | 0.3154 | 0.2727 | 0.2248 | 0.2882 | 0.2531 | 0.2471 | 0.2671 | 0.2875 |
| 52 | 0.3094 | 0.2829 | 0.3132 | 0.2576 | 0.2281 | 0.2843 | 0.2527 | 0.2535 | 0.2718 | 0.2935 |
| 53 | 0.3200 | 0.2891 | 0.3102 | 0.2663 | 0.2299 | 0.2901 | 0.2544 | 0.2516 | 0.2737 | 0.2808 |
| 54 | 0.3169 | 0.2812 | 0.3064 | 0.2541 | 0.2245 | 0.2938 | 0.2518 | 0.2480 | 0.2663 | 0.2860 |
| 55 | 0.3108 | 0.2772 | 0.3070 | 0.2589 | 0.2225 | 0.2883 | 0.2509 | 0.2511 | 0.2687 | 0.2895 |
| 56 | 0.3073 | 0.2812 | 0.3085 | 0.2562 | 0.2302 | 0.2847 | 0.2506 | 0.2554 | 0.2729 | 0.2925 |
| 57 | 0.3191 | 0.2840 | 0.3141 | 0.2609 | 0.2270 | 0.2780 | 0.2489 | 0.2489 | 0.2654 | 0.2944 |
| 58 | 0.3131 | 0.2826 | 0.3111 | 0.2693 | 0.2254 | 0.2910 | 0.2592 | 0.2484 | 0.2616 | 0.2907 |
| 59 | 0.3128 | 0.2874 | 0.3036 | 0.2647 | 0.2275 | 0.2923 | 0.2538 | 0.2574 | 0.2577 | 0.2848 |
| 60 | 0.3065 | 0.2764 | 0.3074 | 0.2604 | 0.2308 | 0.2837 | 0.2553 | 0.2523 | 0.2629 | 0.2896 |

## 4. HPC prediction error per class (mean, per epoch)

| ep | airplane | bird | car | cat | deer | dog | horse | monkey | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7581 | 0.6766 | 0.7393 | 0.6459 | 0.6106 | 0.6619 | 0.6586 | 0.6291 | 0.7203 | 0.7160 |
| 35 | 0.1656 | 0.1514 | 0.1660 | 0.1409 | 0.1353 | 0.1400 | 0.1334 | 0.1398 | 0.1553 | 0.1612 |
| 36 | 0.1652 | 0.1507 | 0.1671 | 0.1409 | 0.1371 | 0.1402 | 0.1335 | 0.1400 | 0.1562 | 0.1613 |
| 37 | 0.1642 | 0.1507 | 0.1670 | 0.1409 | 0.1374 | 0.1406 | 0.1336 | 0.1412 | 0.1562 | 0.1615 |
| 38 | 0.1643 | 0.1506 | 0.1664 | 0.1411 | 0.1373 | 0.1403 | 0.1336 | 0.1415 | 0.1557 | 0.1615 |
| 39 | 0.1641 | 0.1499 | 0.1661 | 0.1408 | 0.1375 | 0.1400 | 0.1338 | 0.1412 | 0.1554 | 0.1616 |
| 40 | 0.1639 | 0.1500 | 0.1662 | 0.1410 | 0.1371 | 0.1401 | 0.1342 | 0.1412 | 0.1550 | 0.1620 |
| 41 | 0.1637 | 0.1501 | 0.1662 | 0.1412 | 0.1370 | 0.1399 | 0.1341 | 0.1412 | 0.1549 | 0.1618 |
| 42 | 0.1634 | 0.1500 | 0.1663 | 0.1415 | 0.1371 | 0.1395 | 0.1338 | 0.1413 | 0.1545 | 0.1617 |
| 43 | 0.1633 | 0.1499 | 0.1660 | 0.1415 | 0.1369 | 0.1393 | 0.1338 | 0.1415 | 0.1545 | 0.1615 |
| 44 | 0.1631 | 0.1499 | 0.1661 | 0.1419 | 0.1367 | 0.1390 | 0.1338 | 0.1414 | 0.1542 | 0.1614 |
| 45 | 0.1631 | 0.1497 | 0.1659 | 0.1418 | 0.1365 | 0.1389 | 0.1337 | 0.1412 | 0.1541 | 0.1615 |
| 46 | 0.1629 | 0.1496 | 0.1656 | 0.1418 | 0.1363 | 0.1389 | 0.1336 | 0.1411 | 0.1540 | 0.1614 |
| 47 | 0.1628 | 0.1498 | 0.1655 | 0.1418 | 0.1361 | 0.1387 | 0.1336 | 0.1408 | 0.1537 | 0.1612 |
| 48 | 0.1626 | 0.1499 | 0.1653 | 0.1417 | 0.1361 | 0.1388 | 0.1335 | 0.1408 | 0.1536 | 0.1610 |
| 49 | 0.1625 | 0.1497 | 0.1651 | 0.1420 | 0.1360 | 0.1387 | 0.1335 | 0.1408 | 0.1536 | 0.1608 |
| 50 | 0.1624 | 0.1497 | 0.1649 | 0.1418 | 0.1359 | 0.1386 | 0.1335 | 0.1405 | 0.1534 | 0.1606 |
| 51 | 0.1623 | 0.1496 | 0.1648 | 0.1417 | 0.1357 | 0.1386 | 0.1333 | 0.1404 | 0.1533 | 0.1605 |
| 52 | 0.1622 | 0.1495 | 0.1648 | 0.1416 | 0.1356 | 0.1386 | 0.1333 | 0.1404 | 0.1533 | 0.1605 |
| 53 | 0.1622 | 0.1494 | 0.1648 | 0.1415 | 0.1355 | 0.1385 | 0.1331 | 0.1402 | 0.1534 | 0.1603 |
| 54 | 0.1621 | 0.1494 | 0.1648 | 0.1415 | 0.1354 | 0.1385 | 0.1331 | 0.1401 | 0.1534 | 0.1602 |
| 55 | 0.1620 | 0.1492 | 0.1647 | 0.1414 | 0.1352 | 0.1385 | 0.1330 | 0.1400 | 0.1532 | 0.1602 |
| 56 | 0.1617 | 0.1493 | 0.1645 | 0.1414 | 0.1353 | 0.1385 | 0.1329 | 0.1398 | 0.1532 | 0.1600 |
| 57 | 0.1616 | 0.1493 | 0.1645 | 0.1413 | 0.1352 | 0.1384 | 0.1329 | 0.1398 | 0.1531 | 0.1601 |
| 58 | 0.1616 | 0.1491 | 0.1645 | 0.1414 | 0.1352 | 0.1385 | 0.1329 | 0.1398 | 0.1531 | 0.1600 |
| 59 | 0.1615 | 0.1491 | 0.1645 | 0.1414 | 0.1351 | 0.1383 | 0.1329 | 0.1398 | 0.1530 | 0.1599 |
| 60 | 0.1614 | 0.1491 | 0.1645 | 0.1414 | 0.1351 | 0.1383 | 0.1329 | 0.1398 | 0.1529 | 0.1600 |

## 5. Truck-rank WATCH series (gate amendment 2026-08-16, non-blocking)

| ep | truck Π_D | rank (top-3) | in top-3 | truck−#2 margin |
|---|---|---|---|---|
| 1 | 0.3345 | 3 | True | -0.0358 |
| 35 | 0.2902 | 5 | False | -0.0315 |
| 36 | 0.2975 | 4 | False | -0.0272 |
| 37 | 0.3017 | 3 | True | -0.0154 |
| 38 | 0.2957 | 4 | False | -0.0168 |
| 39 | 0.2987 | 3 | True | -0.0092 |
| 40 | 0.3035 | 3 | True | -0.0142 |
| 41 | 0.2969 | 3 | True | -0.0203 |
| 42 | 0.2971 | 3 | True | -0.0206 |
| 43 | 0.3013 | 3 | True | -0.0144 |
| 44 | 0.2872 | 5 | False | -0.0254 |
| 45 | 0.2955 | 3 | True | -0.0168 |
| 46 | 0.2980 | 3 | True | -0.0074 |
| 47 | 0.2874 | 5 | False | -0.0236 |
| 48 | 0.2930 | 4 | False | -0.0066 |
| 49 | 0.2909 | 4 | False | -0.0170 |
| 50 | 0.2897 | 3 | True | -0.0168 |
| 51 | 0.2875 | 4 | False | -0.0279 |
| 52 | 0.2935 | 3 | True | -0.0159 |
| 53 | 0.2808 | 5 | False | -0.0294 |
| 54 | 0.2860 | 4 | False | -0.0204 |
| 55 | 0.2895 | 3 | True | -0.0175 |
| 56 | 0.2925 | 3 | True | -0.0148 |
| 57 | 0.2944 | 3 | True | -0.0197 |
| 58 | 0.2907 | 4 | False | -0.0204 |
| 59 | 0.2848 | 5 | False | -0.0188 |
| 60 | 0.2896 | 3 | True | -0.0169 |

The margin **narrowed** through the 0.062/0.094 phases (−0.0285 @ e10 in the earlier session → −0.0169 final): truck converged toward #2 instead of diverging — the WATCH's flag threshold (< −0.05) never tripped. Truck's per-class HPC error was the **lowest** of the car/airplane/truck contenders in every logged epoch.

## 6. Step C — PGD-50 matched eval (THREE-WAY: A baseline / B AIS-v1 / C HPC-only)

5 seeds × 300 samples, eps ∈ {0.0, 0.094}, PGD-50, norm-space (Finding-17 matched convention), baseline `trades_large_baseline`. Run 2026-08-18 under commit `3c50d51` — the registry now declares C's trained checkpoint (previously `None`, which silently dropped C from the first sweep), and both legs use per-cell `--resume` + per-leg HF sync. Numbers are the exact HF-synced CSVs.

### 6.1 Aggregated (mean ± std over seeds)

| checkpoint | eps | Acc% | d′ |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | 0.000 | 49.40±3.48 | 1.8348±0.2745 |
| rhan_next_ais_v1_halting_only | 0.094 | 32.53±1.94 | 0.9745±0.1823 |
| rhan_next_hpc_only | 0.000 | 55.20±3.67 | 1.7775±0.1289 |
| rhan_next_hpc_only | 0.094 | 27.73±2.28 | 0.6147±0.3276 |
| trades_large_baseline | 0.000 | 53.47±2.87 | 1.8737±0.2432 |
| trades_large_baseline | 0.094 | 20.40±1.21 | 0.3251±0.1952 |

Crossover @ eps=0.094 (criterion: diff > 2·σ_comb):

| checkpoint | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | **+12.13** | 4.57 | **CROSSOVER REAL** |
| rhan_next_hpc_only | **+7.33** | 5.16 | **CROSSOVER REAL** |

### 6.2 Per-seed

| checkpoint | seed | eps | Acc% | d′ |
|---|---|---|---|---|
| trades_large_baseline | 41 | 0.000 | 51.00 | 1.9081 |
| trades_large_baseline | 41 | 0.094 | 19.33 | 0.2296 |
| trades_large_baseline | 42 | 0.000 | 56.33 | 2.0828 |
| trades_large_baseline | 42 | 0.094 | 22.00 | 0.1662 |
| trades_large_baseline | 43 | 0.000 | 50.00 | 2.1296 |
| trades_large_baseline | 43 | 0.094 | 21.33 | 0.6586 |
| trades_large_baseline | 44 | 0.000 | 56.00 | 1.6498 |
| trades_large_baseline | 44 | 0.094 | 20.00 | 0.3286 |
| trades_large_baseline | 45 | 0.000 | 54.00 | 1.5981 |
| trades_large_baseline | 45 | 0.094 | 19.33 | 0.2427 |
| rhan_next_ais_v1_halting_only | 41 | 0.000 | 50.33 | 2.0008 |
| rhan_next_ais_v1_halting_only | 41 | 0.094 | 32.00 | 0.6753 |
| rhan_next_ais_v1_halting_only | 42 | 0.000 | 49.00 | 2.0939 |
| rhan_next_ais_v1_halting_only | 42 | 0.094 | 33.00 | 1.0119 |
| rhan_next_ais_v1_halting_only | 43 | 0.000 | 44.67 | 1.3856 |
| rhan_next_ais_v1_halting_only | 43 | 0.094 | 31.00 | 1.0336 |
| rhan_next_ais_v1_halting_only | 44 | 0.000 | 54.33 | 1.7986 |
| rhan_next_ais_v1_halting_only | 44 | 0.094 | 35.67 | 1.1705 |
| rhan_next_ais_v1_halting_only | 45 | 0.000 | 48.67 | 1.8953 |
| rhan_next_ais_v1_halting_only | 45 | 0.094 | 31.00 | 0.9812 |
| rhan_next_hpc_only | 41 | 0.000 | 53.33 | 1.7610 |
| rhan_next_hpc_only | 41 | 0.094 | 25.33 | 0.1050 |
| rhan_next_hpc_only | 42 | 0.000 | 54.67 | 1.7942 |
| rhan_next_hpc_only | 42 | 0.094 | 31.33 | 0.7412 |
| rhan_next_hpc_only | 43 | 0.000 | 53.33 | 1.5683 |
| rhan_next_hpc_only | 43 | 0.094 | 26.67 | 1.0025 |
| rhan_next_hpc_only | 44 | 0.000 | 61.67 | 1.8704 |
| rhan_next_hpc_only | 44 | 0.094 | 28.33 | 0.6510 |
| rhan_next_hpc_only | 45 | 0.000 | 53.00 | 1.8937 |
| rhan_next_hpc_only | 45 | 0.094 | 27.00 | 0.5738 |

## 7. Step C — PGD-100 leg (eps=0.094, masking re-confirmation)

Complete 5-seed run under commit `3c50d51`, 2026-08-18, provenance `report/sweep_stage2_hpc_only_pgd100/eval_provenance.json`.

### 7.1 Aggregated (mean ± std over seeds)

| checkpoint | eps | Acc% | d′ |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | 0.094 | 31.80±1.92 | 0.9665±0.1886 |
| rhan_next_hpc_only | 0.094 | 27.40±2.22 | 0.6057±0.3374 |
| trades_large_baseline | 0.094 | 19.87±1.07 | 0.3097±0.1879 |

Crossover @ eps=0.094 (criterion: diff > 2·σ_comb):

| checkpoint | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | **+11.93** | 4.40 | **CROSSOVER REAL** |
| rhan_next_hpc_only | **+7.53** | 4.92 | **CROSSOVER REAL** |

### 7.2 Per-seed

| checkpoint | seed | Acc% | d′ |
|---|---|---|---|
| trades_large_baseline | 41 | 19.00 | 0.2352 |
| trades_large_baseline | 42 | 21.33 | 0.1366 |
| trades_large_baseline | 43 | 20.67 | 0.6272 |
| trades_large_baseline | 44 | 19.00 | 0.3099 |
| trades_large_baseline | 45 | 19.33 | 0.2398 |
| rhan_next_ais_v1_halting_only | 41 | 31.00 | 0.6529 |
| rhan_next_ais_v1_halting_only | 42 | 32.00 | 1.0103 |
| rhan_next_ais_v1_halting_only | 43 | 31.00 | 1.0397 |
| rhan_next_ais_v1_halting_only | 44 | 35.00 | 1.1580 |
| rhan_next_ais_v1_halting_only | 45 | 30.00 | 0.9715 |
| rhan_next_hpc_only | 41 | 24.67 | 0.0890 |
| rhan_next_hpc_only | 42 | 30.67 | 0.6882 |
| rhan_next_hpc_only | 43 | 26.33 | 1.0278 |
| rhan_next_hpc_only | 44 | 28.00 | 0.6527 |
| rhan_next_hpc_only | 45 | 27.33 | 0.5707 |

### 7.3 Masking analysis (PGD-50 → PGD-100 gap)

Gap ≤ 1.0 pp = genuine robustness (no masking); 1.0–2.5 pp = borderline inconclusive (GPU nondeterminism caveat); >2.5 pp = potential masking.

| checkpoint | acc PGD-50 | acc PGD-100 | gap (pp) | verdict |
|---|---|---|---|---|
| trades_large_baseline | 20.40 | 19.87 | 0.53 | **GENUINE** (no masking) |
| rhan_next_ais_v1_halting_only | 32.53 | 31.80 | 0.73 | **GENUINE** (no masking) |
| rhan_next_hpc_only | 27.73 | 27.40 | 0.33 | **GENUINE** (no masking) |

All three checkpoints show PGD-50→100 gaps well within the genuine band (≤1.0 pp). None exhibit gradient masking. The cross-run GPU nondeterminism caveat (~1.5 pp) does not apply here — all gaps are < 1.0 pp.

## 8. Step C — 8-seed rerun (AIS-v1 vs baseline only, `datasets==4.7.0` pinned)

**Motivation:** The 5-seed PGD-50 run (§6) showed AIS-v1 **+12.13 pp** over baseline (32.53 vs 20.40), while the Stage 1 official 8-seed result was **+8.5 pp**. Investigation ruled out checkpoint drift (SHA-256 identical) and attack-code changes (byte-identical), so the discrepancy was attributed to `datasets` library shuffle non-determinism across Kaggle sessions. The `datasets` version was unpinned (`pip install --quiet datasets`), allowing different shuffled 300-sample subsets for the same seed.

**Protocol:** Pin `datasets==4.7.0` in `requirements.txt`, `cloud_setup/colab_notebook_noesis.py`, `cloud_setup/Kaggle_NOESIS.py`. Re-run AIS-v1 vs baseline at 8 seeds (41–48), PGD-50, ε∈{0.0, 0.094}, 300 samples, batch 32, norm-space. Local GPU (RTX 4060, 8.6 GB VRAM). Commit `707a83d`. Provenance `report/sweep_rerun_ais_v1_8seed/eval_provenance.json`.

### 8.1 Aggregated (mean ± std over 8 seeds)

| checkpoint | eps | Acc% | d′ |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | 0.000 | 50.38±3.09 | 1.8031±0.2654 |
| rhan_next_ais_v1_halting_only | 0.094 | 31.67±2.92 | 0.8447±0.2015 |
| trades_large_baseline | 0.000 | 54.54±2.95 | 1.8364±0.2081 |
| trades_large_baseline | 0.094 | 23.92±4.62 | 0.5942±0.1655 |

Crossover @ eps=0.094 (criterion: diff > 2·σ_comb):

| checkpoint | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|
| rhan_next_ais_v1_halting_only | **+7.75** | 10.93 | positive but **NOT significant** |

### 8.2 Per-seed

| checkpoint | seed | eps | Acc% | d′ |
|---|---|---|---|---|
| rhan_next_ais_v1_halting_only | 41 | 0.000 | 50.33 | 2.0008 |
| rhan_next_ais_v1_halting_only | 41 | 0.094 | 31.00 | 0.8350 |
| rhan_next_ais_v1_halting_only | 42 | 0.000 | 48.67 | 2.0939 |
| rhan_next_ais_v1_halting_only | 42 | 0.094 | 35.67 | 1.1609 |
| rhan_next_ais_v1_halting_only | 43 | 0.000 | 44.67 | 1.3856 |
| rhan_next_ais_v1_halting_only | 43 | 0.094 | 27.00 | 0.6515 |
| rhan_next_ais_v1_halting_only | 44 | 0.000 | 54.33 | 1.7986 |
| rhan_next_ais_v1_halting_only | 44 | 0.094 | 33.33 | 0.9940 |
| rhan_next_ais_v1_halting_only | 45 | 0.000 | 48.33 | 1.8953 |
| rhan_next_ais_v1_halting_only | 45 | 0.094 | 31.00 | 0.6213 |
| rhan_next_ais_v1_halting_only | 46 | 0.000 | 53.00 | 1.8304 |
| rhan_next_ais_v1_halting_only | 46 | 0.094 | 28.33 | 0.6116 |
| rhan_next_ais_v1_halting_only | 47 | 0.000 | 51.33 | 2.0005 |
| rhan_next_ais_v1_halting_only | 47 | 0.094 | 33.00 | 0.9685 |
| rhan_next_ais_v1_halting_only | 48 | 0.000 | 52.33 | 1.4197 |
| rhan_next_ais_v1_halting_only | 48 | 0.094 | 34.00 | 0.9148 |
| trades_large_baseline | 41 | 0.000 | 51.00 | 1.9081 |
| trades_large_baseline | 41 | 0.094 | 25.00 | 0.2166 |
| trades_large_baseline | 42 | 0.000 | 56.33 | 2.0828 |
| trades_large_baseline | 42 | 0.094 | 22.00 | 0.6920 |
| trades_large_baseline | 43 | 0.000 | 50.00 | 2.1296 |
| trades_large_baseline | 43 | 0.094 | 18.33 | 0.6258 |
| trades_large_baseline | 44 | 0.000 | 56.00 | 1.6498 |
| trades_large_baseline | 44 | 0.094 | 22.00 | 0.6980 |
| trades_large_baseline | 45 | 0.000 | 54.00 | 1.5981 |
| trades_large_baseline | 45 | 0.094 | 23.00 | 0.6924 |
| trades_large_baseline | 46 | 0.000 | 53.67 | 1.9409 |
| trades_large_baseline | 46 | 0.094 | 20.67 | 0.6245 |
| trades_large_baseline | 47 | 0.000 | 58.67 | 1.6342 |
| trades_large_baseline | 47 | 0.094 | 27.00 | 0.5094 |
| trades_large_baseline | 48 | 0.000 | 56.67 | 1.7479 |
| trades_large_baseline | 48 | 0.094 | 33.33 | 0.6949 |

### 8.3 Cross-run comparison (AIS-v1 vs baseline @ ε=0.094)

| run | seeds | AIS-v1 Acc% | Baseline Acc% | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|---|---|---|
| Stage 1 official (pre-registered) | 41–48 (8) | 32.21±2.74 | 23.71±3.47 | +8.5 | 8.77 | NOT significant |
| Step C PGD-50 three-way (§6) | 41–45 (5) | 32.53±1.94 | 20.40±1.21 | +12.13 | 4.57 | CROSSOVER REAL |
| **8-seed rerun, `datasets==4.7.0` (§8)** | **41–48 (8)** | **31.67±2.92** | **23.92±4.62** | **+7.75** | **10.93** | **NOT significant** |

The 8-seed rerun with pinned `datasets` closely reproduces the Stage 1 result (+7.75 vs +8.5 pp). The 5-seed Step C run's +12.13 pp was driven by an anomalously low baseline (20.40%, which with 8 seeds recovers to 23.92%). The high baseline variance across runs (σ ≈ 1.2–4.6 pp per seed mean) is the primary source of crossover-estimate instability.

## 9. Step C — 8-seed rerun (HPC-only vs baseline, `datasets==4.7.0` pinned)

**Motivation:** The 5-seed PGD-50/100 runs (§6/§7) showed HPC-only **+7.33/+7.53 pp** over baseline. Parallel to the AIS-v1 8-seed rerun (§8), this may be inflated by baseline variance under unpinned `datasets`. Re-run with `datasets==4.7.0` pinned.

**Protocol:** Same as §8 — 8 seeds (41–48), 300 samples, batch 32, norm-space, local GPU (RTX 4060). PGD-50 and PGD-100 run as separate passes. Commit `707a83d`.

### 9.1 Aggregated (mean ± std over 8 seeds)

**PGD-50:**

| checkpoint | eps | Acc% | d′ |
|---|---|---|---|
| rhan_next_hpc_only | 0.000 | 55.96±2.97 | 1.7474±0.1026 |
| rhan_next_hpc_only | 0.094 | 27.83±2.32 | 0.7126±0.2790 |
| trades_large_baseline | 0.000 | 54.54±2.95 | 1.8364±0.2081 |
| trades_large_baseline | 0.094 | 23.92±4.62 | 0.5942±0.1655 |

**PGD-100:**

| checkpoint | eps | Acc% | d′ |
|---|---|---|---|
| rhan_next_hpc_only | 0.094 | 27.46±2.29 | 0.6660±0.2378 |
| trades_large_baseline | 0.094 | 23.17±4.43 | 0.5344±0.1946 |

Crossover @ eps=0.094 (criterion: diff > 2·σ_comb):

| PGD | HPC-only Acc% | Baseline Acc% | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|---|---|
| PGD-50 | 27.83±2.32 | 23.92±4.62 | **+3.92** | 10.34 | positive but **NOT significant** |
| PGD-100 | 27.46±2.29 | 23.17±4.43 | **+4.29** | 9.98 | positive but **NOT significant** |

### 9.2 Per-seed (PGD-50)

| checkpoint | seed | Acc% | d′ |
|---|---|---|---|
| rhan_next_hpc_only | 41 | 25.00 | 0.3963 |
| rhan_next_hpc_only | 42 | 28.67 | 0.8947 |
| rhan_next_hpc_only | 43 | 25.33 | 0.7298 |
| rhan_next_hpc_only | 44 | 30.33 | 0.7665 |
| rhan_next_hpc_only | 45 | 26.00 | 0.4422 |
| rhan_next_hpc_only | 46 | 27.33 | 1.2002 |
| rhan_next_hpc_only | 47 | 28.67 | 0.4923 |
| rhan_next_hpc_only | 48 | 31.33 | 0.6568 |
| trades_large_baseline | 41 | 25.00 | 0.2166 |
| trades_large_baseline | 42 | 22.00 | 0.6920 |
| trades_large_baseline | 43 | 18.33 | 0.6258 |
| trades_large_baseline | 44 | 22.00 | 0.6980 |
| trades_large_baseline | 45 | 23.00 | 0.6924 |
| trades_large_baseline | 46 | 20.67 | 0.6245 |
| trades_large_baseline | 47 | 27.00 | 0.5094 |
| trades_large_baseline | 48 | 33.33 | 0.6949 |

### 9.3 Per-seed (PGD-100)

| checkpoint | seed | Acc% | d′ |
|---|---|---|---|
| rhan_next_hpc_only | 41 | 24.67 | 0.3763 |
| rhan_next_hpc_only | 42 | 28.67 | 0.8758 |
| rhan_next_hpc_only | 43 | 25.00 | 1.0093 |
| rhan_next_hpc_only | 44 | 30.00 | 0.7467 |
| rhan_next_hpc_only | 45 | 25.67 | 0.4298 |
| rhan_next_hpc_only | 46 | 26.67 | 0.8951 |
| rhan_next_hpc_only | 47 | 28.33 | 0.4801 |
| rhan_next_hpc_only | 48 | 30.67 | 0.6151 |
| trades_large_baseline | 41 | 24.00 | 0.1433 |
| trades_large_baseline | 42 | 21.00 | 0.6425 |
| trades_large_baseline | 43 | 17.67 | 0.6145 |
| trades_large_baseline | 44 | 21.67 | 0.4024 |
| trades_large_baseline | 45 | 22.00 | 0.6858 |
| trades_large_baseline | 46 | 20.33 | 0.6066 |
| trades_large_baseline | 47 | 26.67 | 0.5183 |
| trades_large_baseline | 48 | 32.00 | 0.6618 |

### 9.4 Masking check (PGD-50 → PGD-100 gap)

| checkpoint | PGD-50 | PGD-100 | gap (pp) | verdict |
|---|---|---|---|---|
| rhan_next_hpc_only | 27.83 | 27.46 | −0.37 | **GENUINE** (no masking) |
| trades_large_baseline | 23.92 | 23.17 | −0.75 | **GENUINE** (no masking) |

### 9.5 Cross-run comparison (HPC-only vs baseline @ ε=0.094)

| run | seeds | HPC-only Acc% | Baseline Acc% | diff (pp) | 2·σ_comb | verdict |
|---|---|---|---|---|---|---|
| Step C PGD-50 three-way (§6) | 41–45 (5) | 27.73±2.28 | 20.40±1.21 | +7.33 | 5.16 | CROSSOVER REAL |
| Step C PGD-100 three-way (§7) | 41–45 (5) | 27.40±2.22 | 19.87±1.07 | +7.53 | 4.92 | CROSSOVER REAL |
| **8-seed rerun PGD-50 (§9)** | **41–48 (8)** | **27.83±2.32** | **23.92±4.62** | **+3.92** | **10.34** | **NOT significant** |
| **8-seed rerun PGD-100 (§9)** | **41–48 (8)** | **27.46±2.29** | **23.17±4.43** | **+4.29** | **9.98** | **NOT significant** |

The 8-seed rerun with pinned `datasets` shrinks the crossover from +7.33/+7.53 pp (5 seeds) to +3.92/+4.29 pp (8 seeds) — both NOT significant. The HPC-only model's robustness is stable across runs (27.40–27.83 pp), but the baseline recovers from 20.40/19.87 (5 seeds) to 23.92/23.17 (8 seeds). The same baseline-variance mechanism inflated both the AIS-v1 (§8) and HPC-only (§9) 5-seed crossovers.

**HPC-only's +3.92/+4.29 pp is NOT significant. The HPC-only variant does not reliably cross over the TRADES baseline under attack.**

## 10. Key observations

- Best test acc **56.81%** (set in the epoch-1–34 segment; epochs 35–60 traded 52.4–56.5% around it — best synced to HF).
- HPC prediction error converged 0.4432 (e10, prior session) → 0.1506 (e35) → **0.1492 (e60)**; error-map std pinned ≈0.266, max slowly crept 1.48 → 1.87 — no collapse/explosion (trend check PASS).
- Truck Π_D stayed rank 3 the whole segment (car/airplane contested #1: car held #1 at e35–e40 boundary, airplane at several 0.094-phase epochs); truck−#2 margin closed from −0.0239 (e11, prior) to −0.0169.
- β_dyn stepped up with the curriculum: 1.57 (ε=0.062) → 1.94–1.96 (ε=0.094, min 1.75/max 3.25) — the precision controller responded to the heavier perturbation.
- `frac_halted_any: 0.000`, effective steps pinned at the hard cap 4 — expected for the HPC-only variant (entropy-gated halting is AIS-only).
- Recon MSE kept falling through the 0.094 phase (0.901 → 0.858) — the generative prior keeps improving even under the strongest perturbation.
- **Three-way eval (PGD-50, 5 seeds, §6):** HPC-only (C) has the **best clean accuracy** of the three (55.20±3.67 vs baseline 53.47±2.87, AIS-v1 49.40±3.48) but the **weakest robustness** of the two RHANNext variants at ε=0.094 (27.73±2.28 vs AIS-v1 32.53±1.94). Both variants cross over the TRADES baseline: AIS-v1 **+12.13 pp** (2·σ = 4.57), HPC-only **+7.33 pp** (2·σ = 5.16) — both CROSSOVER REAL. **Superseded by 8-seed reruns (§8, §9).**
- **Three-way eval (PGD-100, 5 seeds, §7):** Results are stable from PGD-50: AIS-v1 **+11.93 pp** (2·σ = 4.40), HPC-only **+7.53 pp** (2·σ = 4.92) — both CROSSOVER REAL. PGD-50→100 gaps are all ≤0.73 pp — **no gradient masking** in any checkpoint. **Superseded by 8-seed reruns (§8, §9).**
- **8-seed rerun AIS-v1 (§8):** With `datasets==4.7.0` pinned, AIS-v1 **+7.75 pp** (2·σ = 10.93) — **NOT significant**. Closely reproduces Stage 1 official +8.5 pp. **The pre-registered Stage 1 verdict (+8.5 pp, NOT significant) remains the official record.**
- **8-seed rerun HPC-only (§9):** With `datasets==4.7.0` pinned, HPC-only **+3.92 pp** PGD-50 (2·σ = 10.34) / **+4.29 pp** PGD-100 (2·σ = 9.98) — both **NOT significant**. The 5-seed +7.33/+7.53 pp was inflated by low baseline draws (20.40/19.87% → 23.92/23.17% at 8 seeds). HPC-only's robustness is stable (27.40–27.83%) but **does not reliably cross over the TRADES baseline**.
- **Both RHANNext variants fail to cross over baseline at 8 seeds.** AIS-v1: +7.75 pp (NOT sig); HPC-only: +3.92/+4.29 pp (NOT sig). The 5-seed numbers were uniformly inflated by baseline variance under unpinned `datasets`. Neither variant provides statistically significant adversarial robustness improvement over the TRADES baseline.
- HPC-only's clean-acc edge over AIS-v1 (+5.8 pp, ≈56.0 vs ≈50.4) does **not** translate to a robustness advantage under attack (27.5 vs 31.7 pp at ε=0.094) — the auxiliary HPC signal helps clean generalization but does not add adversarial margin the way the AIS halting/precision machinery does.
