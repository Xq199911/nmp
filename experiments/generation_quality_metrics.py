"""
Compute BLEU-4 and ROUGE-L between ON and OFF generations saved under experiments/realmodel_out.
Saves JSON and CSV summaries.
"""
from __future__ import annotations
import os
import json
import csv
from typing import List

def tokenize(text: str) -> List[str]:
    return text.strip().split()

def ngram_counts(tokens, n):
    counts = {}
    for i in range(len(tokens)-n+1):
        g = tuple(tokens[i:i+n])
        counts[g] = counts.get(g, 0) + 1
    return counts

def modified_precision(ref_tokens, cand_tokens, n):
    ref_counts = ngram_counts(ref_tokens, n)
    cand_counts = ngram_counts(cand_tokens, n)
    if not cand_counts:
        return 0.0
    clipped = 0
    total = 0
    for g, c in cand_counts.items():
        total += c
        clipped += min(c, ref_counts.get(g, 0))
    return clipped / total if total > 0 else 0.0

def bleu4(ref, cand):
    r = tokenize(ref)
    c = tokenize(cand)
    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = [modified_precision(r, c, i) for i in range(1,5)]
    # geometric mean of precisions
    import math
    if any(p == 0 for p in p_ns):
        gm = 0.0
    else:
        gm = math.exp(sum(w * math.log(p) for w,p in zip(weights, p_ns)))
    # brevity penalty
    ref_len = len(r)
    cand_len = len(c)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / (cand_len+1e-12))
    return bp * gm

def lcs_length(a_tokens, b_tokens):
    A = len(a_tokens); B = len(b_tokens)
    dp = [[0]*(B+1) for _ in range(A+1)]
    for i in range(A-1, -1, -1):
        for j in range(B-1, -1, -1):
            if a_tokens[i] == b_tokens[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def rouge_l(ref, cand):
    r = tokenize(ref); c = tokenize(cand)
    if not r or not c:
        return 0.0
    lcs = lcs_length(r, c)
    prec = lcs / len(c)
    rec = lcs / len(r)
    if prec + rec == 0:
        f = 0.0
    else:
        f = (2 * prec * rec) / (prec + rec)
    return f

def load_gens(path):
    try:
        with open(os.path.join(path, "generations.json"), "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj.get("generations", [])
    except Exception:
        return []

def main(on_dir="experiments/realmodel_out/inject_on_attn", off_dir="experiments/realmodel_out/inject_off_attn"):
    on = load_gens(on_dir)
    off = load_gens(off_dir)
    out = []
    for i in range(min(len(on), len(off))):
        a = on[i]["generation"]
        b = off[i]["generation"]
        b4 = bleu4(b, a)  # treat OFF as reference, ON as candidate
        rL = rouge_l(b, a)
        out.append({"prompt": on[i]["prompt"], "bleu4": b4, "rougeL": rL, "len_on": len(a), "len_off": len(b)})
    # save
    out_path = "experiments/realmodel_out/generation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    csv_path = "experiments/realmodel_out/generation_metrics.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["prompt", "bleu4", "rougeL", "len_on", "len_off"])
        for r in out:
            w.writerow([r["prompt"], r["bleu4"], r["rougeL"], r["len_on"], r["len_off"]])
    print("Wrote generation metrics to", out_path, csv_path)

if __name__ == "__main__":
    main()


