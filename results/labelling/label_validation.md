# Label validation (n=30 hand-labelled responses)

## Heuristic vs you (4 classes): accuracy 80.0%, Cohen's kappa 0.723

Confusion matrix (rows = your label, columns = heuristic):
| you \ heuristic | correct | false | mixed | neither |
|---|---|---|---|---|
| correct | 4 | 0 | 2 | 1 |
| false | 0 | 9 | 0 | 1 |
| mixed | 0 | 0 | 3 | 0 |
| neither | 1 | 1 | 0 | 8 |

## The attack-success label ('false' vs everything else)
| Labeller | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Heuristic | 9 | 1 | 1 | 0.90 | 0.90 |
| LLM judge | 9 | 0 | 1 | 1.00 | 0.90 |
(Precision < 1 means the labeller INFLATES ASR; recall < 1 means it DEFLATES it.)

## Pool-wide share of 'false' answers (unique responses to attacked queries, N=287)
- by heuristic: 34.5%
- estimated from your labels (re-weighted by class size): 35.9%
With ~30 labels this estimate is rough (several points of uncertainty); use it as a sanity check, not as a corrected ASR.