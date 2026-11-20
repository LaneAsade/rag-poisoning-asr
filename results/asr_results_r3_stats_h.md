Labels: heuristic

## Attack: contradict (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 20 | 13 | 7 | 10 | 0 |
| B | 17 | 7 | 4 | 20 | 2 |
| G | 16 | 7 | 6 | 4 | 17 |
| R | 4 | 2 | 1 | 26 | 17 |
| R2 | 4 | 0 | 0 | 22 | 24 |
| R3 | 5 | 0 | 0 | 23 | 22 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 13/50 | 7/50 | -12.0 pp [-22.0, -4.0] | 6 | 0 | 0.0312 | 0.0625 |
| G | 13/50 | 7/50 | -12.0 pp [-22.0, -4.0] | 6 | 0 | 0.0312 | 0.0625 |
| R | 13/50 | 2/50 | -22.0 pp [-36.0, -10.0] | 12 | 1 | 0.0034 | 0.0103 |
| R2 | 13/50 | 0/50 | -26.0 pp [-38.0, -14.0] | 13 | 0 | 0.0002 | 0.0012 |
| R3 | 13/50 | 0/50 | -26.0 pp [-38.0, -14.0] | 13 | 0 | 0.0002 | 0.0012 |

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
| A | 0/50 | 40/50 | +0.0 |
| B | 3/50 | 34/50 | -12.0 |
| G | 5/50 | 37/50 | -6.0 |
| R | 7/50 | 34/50 | -12.0 |
| R2 | 8/50 | 33/50 | -14.0 |
| R3 | 8/50 | 33/50 | -14.0 |

## Attack: inject (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 3 | 39 | 0 | 8 | 0 |
| B | 5 | 1 | 1 | 27 | 16 |
| G | 0 | 1 | 0 | 1 | 48 |
| R | 4 | 1 | 0 | 23 | 22 |
| R2 | 4 | 0 | 0 | 22 | 24 |
| R3 | 5 | 0 | 1 | 26 | 18 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 39/50 | 1/50 | -76.0 pp [-88.0, -64.0] | 38 | 0 | 0.0000 | 0.0000 |
| G | 39/50 | 1/50 | -76.0 pp [-88.0, -64.0] | 38 | 0 | 0.0000 | 0.0000 |
| R | 39/50 | 1/50 | -76.0 pp [-88.0, -64.0] | 38 | 0 | 0.0000 | 0.0000 |
| R2 | 39/50 | 0/50 | -78.0 pp [-88.0, -66.0] | 39 | 0 | 0.0000 | 0.0000 |
| R3 | 39/50 | 0/50 | -78.0 pp [-88.0, -66.0] | 39 | 0 | 0.0000 | 0.0000 |

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
| A | 0/50 | 39/50 | +0.0 |
| B | 7/50 | 34/50 | -10.0 |
| G | 7/50 | 36/50 | -6.0 |
| R | 7/50 | 34/50 | -10.0 |
| R2 | 8/50 | 33/50 | -12.0 |
| R3 | 8/50 | 33/50 | -12.0 |

## Attack: replace (attacked n=50, untouched n=50)

### 1. Label distribution on attacked queries
| Condition | correct | false | mixed | neither | blocked |
|---|---|---|---|---|---|
| A | 0 | 39 | 1 | 10 | 0 |
| B | 0 | 38 | 1 | 11 | 0 |
| G | 0 | 38 | 0 | 9 | 3 |
| R | 1 | 19 | 0 | 21 | 9 |
| R2 | 4 | 1 | 0 | 24 | 21 |
| R3 | 5 | 1 | 0 | 25 | 19 |

### 2. Paired ASR comparison vs A (no defense)
| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| B | 39/50 | 38/50 | -2.0 pp [-6.0, +0.0] | 1 | 0 | 1.0000 | 1.0000 |
| G | 39/50 | 38/50 | -2.0 pp [-6.0, +0.0] | 1 | 0 | 1.0000 | 1.0000 |
| R | 39/50 | 19/50 | -40.0 pp [-54.0, -26.0] | 20 | 0 | 0.0000 | 0.0000 |
| R2 | 39/50 | 1/50 | -76.0 pp [-88.0, -64.0] | 38 | 0 | 0.0000 | 0.0000 |
| R3 | 39/50 | 1/50 | -76.0 pp [-88.0, -64.0] | 38 | 0 | 0.0000 | 0.0000 |

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
| A | 0/50 | 41/50 | +0.0 |
| B | 0/50 | 35/50 | -12.0 |
| G | 4/50 | 39/50 | -4.0 |
| R | 3/50 | 36/50 | -10.0 |
| R2 | 8/50 | 33/50 | -16.0 |
| R3 | 8/50 | 33/50 | -16.0 |

## 5. Heuristic vs LLM-judge agreement
n=1123, raw agreement=68.74%, Cohen's kappa=0.445
(Heuristic uses 'mixed'; the judge never does, so some disagreement is structural.)