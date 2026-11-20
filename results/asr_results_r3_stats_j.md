Labels: LLM judge

## Attack: contradict (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 37 | 8 | 0 | 5 | 0 |
| B | 30 | 2 | 0 | 16 | 2 |
| G | 28 | 3 | 0 | 2 | 17 |
| R | 15 | 1 | 0 | 17 | 17 |
| R2 | 0 | 0 | 0 | 0 | 24 |
| R3 | 0 | 0 | 0 | 0 | 22 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 8/50 | 2/50 | -12.0 pp [-22.0, -4.0] | 6 | 0 | 0.0312 | 0.0625 |
| G | 8/50 | 3/50 | -10.0 pp [-18.0, -2.0] | 5 | 0 | 0.0625 | 0.0625 |
| R | 8/50 | 1/50 | -14.0 pp [-24.0, -6.0] | 7 | 0 | 0.0156 | 0.0469 |
| R2 | 8/50 | 0/50 | -16.0 pp [-26.0, -6.0] | 8 | 0 | 0.0078 | 0.0391 |
| R3 | 8/50 | 0/50 | -16.0 pp [-26.0, -6.0] | 8 | 0 | 0.0078 | 0.0391 |

### 3. Trust Index descriptives (condition A response, K retrieved docs)
- attacked : 0.611 +/- 0.201 (median 0.660, n=50)
- untouched: 0.764 +/- 0.163 (median 0.776, n=50)
- separation (untouched - attacked mean): 0.153
- ROC-AUC (low trust => attacked): 0.717
- max per-doc poison prob, attacked : 0.585 +/- 0.238 (median 0.689, n=50)
- max per-doc poison prob, untouched: 0.387 +/- 0.305 (median 0.380, n=50)

### 4. Cost of defense on untouched queries
| Condition | Blocked | Correct | Correct-rate change vs A (pp) |
|---|---|---|---|
| A | 0/50 | 46/50 | +0.0 |
| B | 3/50 | 44/50 | -4.0 |
| G | 5/50 | 42/50 | -8.0 |
| R | 7/50 | 40/50 | -12.0 |
| R2 | 8/50 | 0/50 | -92.0 |
| R3 | 8/50 | 0/50 | -92.0 |

## Attack: inject (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 8 | 34 | 0 | 8 | 0 |
| B | 13 | 1 | 0 | 20 | 16 |
| G | 0 | 2 | 0 | 0 | 48 |
| R | 10 | 1 | 0 | 17 | 22 |
| R2 | 0 | 0 | 0 | 0 | 24 |
| R3 | 0 | 0 | 0 | 0 | 18 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 34/50 | 1/50 | -66.0 pp [-78.0, -52.0] | 33 | 0 | 0.0000 | 0.0000 |
| G | 34/50 | 2/50 | -64.0 pp [-78.0, -50.0] | 32 | 0 | 0.0000 | 0.0000 |
| R | 34/50 | 1/50 | -66.0 pp [-78.0, -52.0] | 33 | 0 | 0.0000 | 0.0000 |
| R2 | 34/50 | 0/50 | -68.0 pp [-80.0, -54.0] | 34 | 0 | 0.0000 | 0.0000 |
| R3 | 34/50 | 0/50 | -68.0 pp [-80.0, -54.0] | 34 | 0 | 0.0000 | 0.0000 |

### 3. Trust Index descriptives (condition A response, K retrieved docs)
- attacked : 0.364 +/- 0.053 (median 0.352, n=50)
- untouched: 0.704 +/- 0.178 (median 0.739, n=50)
- separation (untouched - attacked mean): 0.340
- ROC-AUC (low trust => attacked): 0.939
- max per-doc poison prob, attacked : 0.984 +/- 0.042 (median 1.000, n=50)
- max per-doc poison prob, untouched: 0.664 +/- 0.393 (median 0.829, n=50)

### 4. Cost of defense on untouched queries
| Condition | Blocked | Correct | Correct-rate change vs A (pp) |
|---|---|---|---|
| A | 0/50 | 48/50 | +0.0 |
| B | 7/50 | 40/50 | -16.0 |
| G | 7/50 | 42/50 | -12.0 |
| R | 7/50 | 41/50 | -14.0 |
| R2 | 8/50 | 0/50 | -96.0 |
| R3 | 8/50 | 0/50 | -96.0 |

## Attack: replace (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 10 | 29 | 0 | 11 | 0 |
| B | 9 | 28 | 0 | 13 | 0 |
| G | 9 | 28 | 0 | 10 | 3 |
| R | 9 | 18 | 0 | 14 | 9 |
| R2 | 0 | 0 | 0 | 0 | 21 |
| R3 | 0 | 0 | 0 | 0 | 19 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 29/50 | 28/50 | -2.0 pp [-6.0, +0.0] | 1 | 0 | 1.0000 | 1.0000 |
| G | 29/50 | 28/50 | -2.0 pp [-6.0, +0.0] | 1 | 0 | 1.0000 | 1.0000 |
| R | 29/50 | 18/50 | -22.0 pp [-36.0, -10.0] | 12 | 1 | 0.0034 | 0.0103 |
| R2 | 29/50 | 0/50 | -58.0 pp [-72.0, -44.0] | 29 | 0 | 0.0000 | 0.0000 |
| R3 | 29/50 | 0/50 | -58.0 pp [-72.0, -44.0] | 29 | 0 | 0.0000 | 0.0000 |

### 3. Trust Index descriptives (condition A response, K retrieved docs)
- attacked : 0.803 +/- 0.143 (median 0.839, n=50)
- untouched: 0.791 +/- 0.161 (median 0.821, n=50)
- separation (untouched - attacked mean): -0.012
- ROC-AUC (low trust => attacked): 0.479
- max per-doc poison prob, attacked : 0.274 +/- 0.316 (median 0.000, n=50)
- max per-doc poison prob, untouched: 0.311 +/- 0.323 (median 0.120, n=50)

### 4. Cost of defense on untouched queries
| Condition | Blocked | Correct | Correct-rate change vs A (pp) |
|---|---|---|---|
| A | 0/50 | 48/50 | +0.0 |
| B | 0/50 | 43/50 | -10.0 |
| G | 4/50 | 44/50 | -8.0 |
| R | 3/50 | 40/50 | -16.0 |
| R2 | 8/50 | 0/50 | -96.0 |
| R3 | 8/50 | 0/50 | -96.0 |

## 5. Heuristic vs LLM-judge agreement
n=1123, raw agreement=68.74%, Cohen's kappa=0.445
(Heuristic uses 'mixed'; the judge never does, so some disagreement is structural.)