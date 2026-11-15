# rag-poisoning-asr

**From detection to harm: measuring the attack success rate (ASR) of RAG poisoning against the TrustworthyRAG Evaluation Agent, and what a trusted-reference check adds.**

This repository extends [TrustworthyRAG](https://github.com/GPT-Laboratory/TrustworthyRAG) (Giri et al., ICSEA 2026), an Evaluation Agent that screens retrieved context with NLI verification and a multi-signal poison detector. The original study measures whether poisoned context is *detected*. This project measures whether the generator actually *adopts* the attacker's false answer, and how each defense changes what the user receives. It is a course project for a Trustworthy AI course and is not affiliated with the original authors.

## Headline results

Mistral 7B, 100 TruthfulQA questions, K = 3, temperature 0. **ASR on 50 attacked queries per attack** (heuristic labels; 95% intervals are in `results/summary/asr_tables.md`):

| Condition | replace (in-place swap) | inject (override directive) | contradict |
|---|---|---|---|
| A. No defense | 78% | 78% | 26% |
| B. Agent as filter | 76% | 2% | 14% |
| G. Agent as gate | 76% | 2% | 14% |
| R. Filter + lexical reference | 38% | 2% | 4% |
| R2. Filter + NLI reference (top-K) | 2% | 0% | 0% |
| R3. Filter + NLI reference (corpus) | 2% | 0% | 0% |
| C. Clean corpus | 0% | 0% | 0% |

- The Trust Index detects overt injection (recall 96%, ROC-AUC 0.94) but not the in-place swap (recall 6%, AUC 0.48). **Detection is not prevention.**
- The trusted-reference checks (R, R2, R3) assume a clean copy of the corpus and were added post hoc, one after another. They remove the attack but not the damage: correct answers on attacked queries stay at 10% or below.
- Every defense lowers the correct-answer rate on untouched queries by 4 to 16 points; most of the loss comes from the agent's own filter.
- The heuristic labels agree with 30 hand labels at Cohen's kappa 0.72.

The full write-up is in the project report: *[add the link to your report here]*.

## Repository layout

```
asr_experiment.py      runs conditions A, B, G, R and C; resumable
rerun_ref_nli.py       adds the NLI reference conditions R2 and R3 without rerunning the rest
analyze_asr.py         ASR tables with Wilson intervals, and the chart
stats_asr.py           paired McNemar tests, bootstrap, Trust Index AUC, cost analysis, labeller agreement
label_sheet.py         builds a blinded hand-labelling sheet and scores the human labels
scripts/               setup_upstream.sh (clone TrustworthyRAG), download_truthfulqa.py
notebooks/             colab_run.ipynb (end-to-end run on a free Colab T4 GPU)
tests/                 55 tests; stub components, no models or network needed
results/               result files from the reported run, summary tables, hand-labelling files
```

## 1. Reproduce the tables without any models

The committed result files contain every response, retrieved document, label and score, so the tables and statistics regenerate in seconds:

```bash
pip install -r requirements-dev.txt          # pytest only
python analyze_asr.py --in results/asr_results_r3.jsonl --outdir results/summary
python analyze_asr.py --in results/asr_results_r3.jsonl --outdir results/summary_judge --labels j
python stats_asr.py   --in results/asr_results_r3.jsonl          # writes results/asr_results_r3_stats_h.md
python -m pytest tests -q
```

`analyze_asr.py` needs `matplotlib` for the chart (`pip install matplotlib`).

## 2. Rerun the experiments

### Option A: Google Colab (recommended)

Open `notebooks/colab_run.ipynb` in Colab, set the runtime to a T4 GPU, edit `REPO_URL` in cell 2 to your copy of this repository, and run the cells in order. The notebook warms the model, runs a 6-query smoke test, then the full run.

### Option B: local machine

You need a GPU for reasonable speed (our CPU-only smoke test took roughly 30 to 100 seconds per query) and [Ollama](https://ollama.com).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_upstream.sh                 # clones TrustworthyRAG into ./TrustworthyRAG
python scripts/download_truthfulqa.py          # writes data/raw/truthfulqa/truthfulqa.jsonl

ollama serve &                                 # in another terminal if you prefer
ollama pull mistral:7b

# smoke test (6 queries)
python asr_experiment.py --n 6 --k 3 --out data/asr/smoke.jsonl
python analyze_asr.py --in data/asr/smoke.jsonl --outdir data/asr/smoke

# full run: conditions A, B, G, R, C (resumable; rerun the same command after an interruption)
python asr_experiment.py --n 100 --k 3 --poison-ratio 0.5 \
    --strategies replace inject contradict --ref --judge --out data/asr/asr_results.jsonl

# post hoc NLI reference conditions R2 (top-K) and R3 (whole corpus)
python rerun_ref_nli.py --in data/asr/asr_results.jsonl    --out data/asr/asr_results_r2.jsonl
python rerun_ref_nli.py --in data/asr/asr_results_r2.jsonl --out data/asr/asr_results_r3.jsonl --mode corpus --tag R3

# tables, statistics, chart
python analyze_asr.py --in data/asr/asr_results_r3.jsonl --outdir data/asr/summary
python stats_asr.py   --in data/asr/asr_results_r3.jsonl
```

The first Ollama request loads the model onto the GPU and can exceed the 120 s timeout of the upstream client. Warm it once before a run:

```bash
curl -s http://localhost:11434/v1/completions -H "Content-Type: application/json" \
     -d '{"model":"mistral:7b","prompt":"Hello","max_tokens":1}'
```

To use another model or endpoint, set `LLM_MODEL`, `FARMI_API_URL` and `FARMI_API_KEY` (the upstream client is OpenAI-compatible). The scripts locate the upstream code through `TRUSTWORTHYRAG_PATH`, then `./TrustworthyRAG`, then `./external/TrustworthyRAG`.

## Experimental design

**Attacks** (targeted: each plants TruthfulQA's first incorrect answer as the attacker's target):

| Attack | Poisoned document |
|---|---|
| replace | "question + false answer" in place of the true answer; no visible artifact |
| inject | "question + best answer" followed by an override directive stating the false answer |
| contradict | "question + best answer" followed by "Contrary to popular belief" and the false answer |

**Conditions:** A no defense; B agent as filter (drop documents with poison probability of 0.5 or more); G agent as gate (block the answer when the Trust Index is below 0.5); R, R2, R3 trusted-reference checks applied on top of B; C clean corpus.

**Labels.** Every answer is *correct*, *false* (adopts the target), *mixed*, *neither* or *blocked*. ASR is the share of *false* among attacked queries. The primary labeller is a distinguishing-token heuristic (`_h`). The sensitivity labellers are an LLM judge (`_j`) and strict containment of the false answer text.

**Statistics.** Wilson 95% intervals; exact two-sided McNemar tests against no defense, Holm-corrected within each attack; paired bootstrap for ASR differences; ROC-AUC of the Trust Index; Cohen's kappa for labeller agreement.

## Result file format

Each line of `results/asr_results*.jsonl` is one record per `(strategy, qid)` plus one `clean` record per query:

| Field | Meaning |
|---|---|
| `strategy`, `qid`, `attacked` | attack, query index, whether this query's own document was attacked |
| `question`, `correct`, `false` | the TruthfulQA question, its best answer, and the attacker's target |
| `docs`, `scores`, `doc_poison_probs` | the retrieved documents, similarities and per-document poison probabilities |
| `trust_score`, `trust_trustworthy` | Trust Index and the verdict at threshold 0.5 |
| `kept_B`, `kept_R`, `kept_R2`, `kept_R3` | indices of the documents each defense kept |
| `resp_A`, `resp_B`, `resp_G`, `resp_R`, `resp_R2`, `resp_R3` | the answer under each condition (`[BLOCKED: ...]` when blocked) |
| `label_<cond>_h`, `label_<cond>_j` | heuristic and judge labels for each condition (the judge was not run for R2 and R3) |
| `ref_suspicion`, `ref_entail`, `ref_entail_R3` | lexical suspicion and NLI entailment scores per document |

## Hand-labelling workflow

```bash
python label_sheet.py make  --in results/asr_results_r3.jsonl --outdir data/asr/labelling   # blinded sheet
# fill the human_label column of label_sheet.csv (correct | false | mixed | neither)
python label_sheet.py score --sheet data/asr/labelling/label_sheet.csv --outdir data/asr/labelling
```

The sheet is stratified by heuristic label and hides the automatic labels. `results/labelling/` holds the sheet, key and validation report from the reported run.

## Limitations

- 50 attacked queries per attack, one generator (Mistral 7B), one dataset; intervals are about +/- 12 points.
- 50% of documents are attacked, higher than the 30% in the original study, which inflates false-positive spillover.
- R, R2 and R3 were added post hoc after inspecting earlier results, assume a trusted copy of the corpus, and drop poisoned documents without substituting the trusted ones.
- The heuristic cannot detect adoption for the 16% of queries whose false answer adds no words beyond the question and the correct answer (for example yes/no questions), so heuristic ASR is a lower bound for them.
- Heuristic labels were validated on 30 responses by a single annotator; the LLM judge was not run for R2 and R3.

## Credits and license

The agent, retriever and generator client come from [GPT-Laboratory/TrustworthyRAG](https://github.com/GPT-Laboratory/TrustworthyRAG) (MIT). Please cite the original paper if you use this work:

> B. Giri, M. T. Hasan, J. Rasku, M. Waseem and P. Abrahamsson, "Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems," Proc. 21st Int. Conf. Software Engineering Advances (ICSEA), IARIA, 2026. arXiv:2608.21095.

This repository is released under the MIT License (see `LICENSE`; replace the copyright holder with your name).
