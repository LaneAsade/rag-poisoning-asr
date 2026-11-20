"""
rerun_ref_nli.py -- replace the weak lexical trusted-reference check (condition R)
with an NLI-entailment check (condition R2), WITHOUT rerunning anything else.

Put next to asr_experiment.py in the repo root. Needs Ollama running (for the few
new generations) and a GPU for fast NLI.

    python rerun_ref_nli.py --in data/asr/asr_results.jsonl --out data/asr/asr_results_r2.jsonl

Method (fixed BEFORE looking at any R2 output):
  trusted docs  = the clean-corpus top-K already stored in the 'clean' records
  entail(d)     = max over trusted docs t of P(entailment | premise=t, hypothesis=d)   [bart-large-mnli]
  R2 keeps a retrieved doc iff the agent's filter (B) kept it AND entail(d) >= --entail-threshold (default 0.5)
  Answers are regenerated only when the kept-doc set differs from A, B or the old R.

Variant R3 (--mode corpus --tag R3): the trusted set is the WHOLE clean corpus instead of the stored
top-K. For each retrieved doc, the --candidates (default 5) most word-similar clean docs are scored by NLI
and the max entailment is used. Run it on the R2 output so both conditions end up in one file:
    python rerun_ref_nli.py --in data/asr/asr_results_r2.jsonl --out data/asr/asr_results_r3.jsonl --mode corpus --tag R3

Disclosure for the report: R (lexical, threshold 0.3) was run first and under-performed;
R2 was added afterwards. The threshold is the NLI default (0.5), not tuned on outcomes.
The script also prints threshold-free document-level AUCs for both checks.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import asr_experiment as ae
import stats_asr as sa


def pair_entail(nli, cache, premise, hypothesis):
    key = (premise, hypothesis)
    if key not in cache:
        cache[key] = float(nli(premise, hypothesis))
    return cache[key]


def ae_norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def poisoned_doc_indices(rec):
    """A retrieved doc counts as attacked if it contains the planted false answer (normalised)."""
    if not rec["attacked"]:
        return []
    return [i for i, d in enumerate(rec["docs"]) if ae_norm(rec["false"]) in ae_norm(d)]


def corpus_from_clean(records):
    """Rebuild the clean knowledge base ('question + best answer') from the clean records."""
    seen, corpus = set(), []
    for r in records:
        if r["strategy"] == "clean":
            d = f"{r['question']} {r['correct']}"
            if d not in seen:
                seen.add(d)
                corpus.append(d)
    return corpus


def top_candidates(doc, corpus, corpus_toks, m):
    """The m clean docs with the highest word-set Jaccard overlap with `doc`."""
    dt = ae.toks(doc)
    scored = []
    for t, tt in zip(corpus, corpus_toks):
        union = len(dt | tt)
        scored.append((len(dt & tt) / union if union else 0.0, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:m]]


def process(records, nli, gen, judge, args, tag="R2", mode="topk", candidates=5):
    """
    records : list of dicts from an asr results file
    nli     : callable(premise, hypothesis) -> entailment probability
    gen     : callable(question, docs) -> response text
    judge   : callable(prompt) -> text, or None
    mode    : 'topk'   trusted docs = clean top-K stored for the same query (R2)
              'corpus' trusted docs = best-matching docs from the whole clean corpus (R3)
    Fields written: ref_entail (R2) or ref_entail_<tag>, kept_<tag>, resp_<tag>, label_<tag>_h/_j
    """
    clean = {r["qid"]: r for r in records if r["strategy"] == "clean"}
    corpus = corpus_from_clean(records)
    corpus_toks = [ae.toks(t) for t in corpus]
    ent_key = "ref_entail" if tag == "R2" else f"ref_entail_{tag}"
    cache, gen_cache, cand_cache, out = {}, {}, {}, []

    def label(resp, rec):
        it = {"question": rec["question"], "correct": rec["correct"], "false": rec["false"]}
        h, j = ae.label_response(resp, it, None)
        if resp != ae.BLOCKED and judge is not None:
            j = ae.parse_judge(judge(ae.judge_prompt(rec["question"], resp, rec["correct"], rec["false"])))
        return h, j

    def trusted_for(rec, doc):
        if mode == "topk":
            return clean[rec["qid"]]["docs"]
        if doc not in cand_cache:
            cand_cache[doc] = top_candidates(doc, corpus, corpus_toks, candidates)
        return cand_cache[doc]

    for rec in records:
        if rec["strategy"] == "clean":
            out.append(rec)
            continue
        ent = [max((pair_entail(nli, cache, t, d) for t in trusted_for(rec, d)), default=0.5)
               for d in rec["docs"]]
        keep_b = rec["kept_B"]
        keep = [i for i in keep_b if ent[i] >= args.entail_threshold]

        docs = rec["docs"]
        resp = None
        if len(keep) == len(docs):
            resp = rec["resp_A"]
        elif not keep:
            resp = ae.BLOCKED
        else:  # reuse any earlier condition that kept exactly the same docs
            for k, v in rec.items():
                if k.startswith("kept_") and k != f"kept_{tag}" and v == keep and f"resp_{k[5:]}" in rec:
                    resp = rec[f"resp_{k[5:]}"]
                    break
        if resp is None:
            key = (rec["question"], tuple(keep), tuple(docs[i] for i in keep))
            if key not in gen_cache:
                gen_cache[key] = gen(rec["question"], [docs[i] for i in keep])
            resp = gen_cache[key]

        new = dict(rec)
        new[ent_key] = ent
        new[f"kept_{tag}"] = keep
        new[f"resp_{tag}"] = resp
        new[f"label_{tag}_h"], new[f"label_{tag}_j"] = label(resp, rec)
        out.append(new)
    return out


def doc_level_report(out, key="ref_entail"):
    """
    Threshold-free document-level check. Ground truth: a retrieved doc is POISONED iff its text is not
    in the clean corpus. (This includes poisoned neighbours retrieved by queries that were not attacked.)
    AUC = chance that a poisoned doc gets a lower entailment (more suspicious) than a clean one.
    """
    corpus = set(corpus_from_clean(out))
    lines = ["\nDocument-level discrimination (poisoned = text not in the clean corpus; 0.5 = chance)",
             "| Attack | poisoned docs | clean docs | mean entailment (poisoned / clean) | AUC NLI check | AUC old lexical check |",
             "|---|---|---|---|---|---|"]
    for s_ in sorted({r["strategy"] for r in out if r["strategy"] != "clean"}):
        pe, ce, ps, cs = [], [], [], []
        for r in out:
            if r["strategy"] != s_:
                continue
            for i, d in enumerate(r["docs"]):
                bad = d not in corpus
                (pe if bad else ce).append(r[key][i])
                if "ref_suspicion" in r:
                    (ps if bad else cs).append(r["ref_suspicion"][i])
        if not pe or not ce:
            continue
        auc_nli = sa.auc_lower_is_positive(pe, ce)
        auc_lex = sa.auc_lower_is_positive([-x for x in ps], [-x for x in cs]) if ps and cs else float("nan")
        mean = lambda v: sum(v) / len(v)
        lines.append(f"| {s_} | {len(pe)} | {len(ce)} | {mean(pe):.2f} / {mean(ce):.2f} | {auc_nli:.3f} | {auc_lex:.3f} |")
    return "\n".join(lines)


class Real:
    def __init__(self, args):
        sys.path.insert(0, os.getcwd())
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        import torch
        from src.evaluation_agent import NLIVerifier
        from src.generator import Generator
        cfg = {
            "LLM_PROVIDER": "openai_compatible",
            "FARMI_API_URL": os.environ.get("FARMI_API_URL", "http://localhost:11434/v1"),
            "FARMI_API_KEY": os.environ.get("FARMI_API_KEY", "ollama"),
            "LLM_MODEL": os.environ.get("LLM_MODEL", "mistral:7b"),
            "MAX_NEW_TOKENS": args.max_tokens,
            "TEMPERATURE": args.temperature,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"NLI device: {device} | LLM: {cfg['LLM_MODEL']}")
        self.verifier = NLIVerifier(model_name="facebook/bart-large-mnli", device=device)
        self.generator = Generator(config=cfg)
        self._warm_up(cfg)

    @staticmethod
    def _warm_up(cfg):
        """The first request loads the model onto the GPU and can exceed the repo's 120 s timeout."""
        import requests
        print("Warming up the LLM (first load can take a few minutes)...", flush=True)
        try:
            requests.post(
                cfg["FARMI_API_URL"].rstrip("/") + "/completions",
                headers={"Authorization": f"Bearer {cfg['FARMI_API_KEY']}"},
                json={"model": cfg["LLM_MODEL"], "prompt": "Hello", "max_tokens": 1},
                timeout=900,
            )
            print("LLM ready.", flush=True)
        except Exception as e:  # keep going; generation retries below will report real problems
            print(f"Warm-up failed ({e!r}); is Ollama running and the model pulled?", flush=True)

    def nli(self, premise, hypothesis):
        return self.verifier.verify_pair(premise, hypothesis).entailment_score

    def gen(self, question, docs, attempts=3):
        import time
        for k in range(attempts):
            try:
                return self.generator.generate(question=question, context_docs=docs)
            except Exception as e:
                if k == attempts - 1:
                    raise
                print(f"Generation failed ({type(e).__name__}); retry {k + 1}/{attempts - 1}", flush=True)
                time.sleep(5)

    def judge(self, prompt):
        return self.generator.llm_client.generate(prompt, max_tokens=8, temperature=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/asr/asr_results.jsonl")
    ap.add_argument("--out", default="data/asr/asr_results_r2.jsonl")
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    ap.add_argument("--mode", choices=["topk", "corpus"], default="topk")
    ap.add_argument("--tag", default="R2", help="condition name written to the output (R2, R3, ...)")
    ap.add_argument("--candidates", type=int, default=5, help="corpus mode: clean docs scored per retrieved doc")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=150)
    args = ap.parse_args()

    args.inp = str(Path(args.inp).resolve())     # resolve before changing directory
    args.out = str(Path(args.out).resolve())
    os.chdir(ae.find_repo_root())
    records = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} records. Condition {args.tag} ({args.mode}): keep doc iff B keeps it "
          f"AND entailment >= {args.entail_threshold}")
    real = Real(args)
    out = process(records, real.nli, real.gen, real.judge if args.judge else None, args,
                  tag=args.tag, mode=args.mode, candidates=args.candidates)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")
    print(doc_level_report(out, "ref_entail" if args.tag == "R2" else f"ref_entail_{args.tag}"))


if __name__ == "__main__":
    main()
