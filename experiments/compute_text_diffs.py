import json
import difflib
import os


def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    base = "experiments/realmodel_out"
    on_path = os.path.join(base, "inject_on", "generations.json")
    off_path = os.path.join(base, "inject_off", "generations.json")
    jo = json.load(open(on_path, "r", encoding="utf-8"))
    jf = json.load(open(off_path, "r", encoding="utf-8"))
    go = [g["generation"] for g in jo.get("generations", [])]
    gf = [g["generation"] for g in jf.get("generations", [])]
    metrics = []
    for i, (a, b) in enumerate(zip(go, gf)):
        r = difflib.SequenceMatcher(None, a, b).ratio()
        j = jaccard(a, b)
        ld = abs(len(a) - len(b))
        metrics.append({"idx": i, "ratio": r, "jaccard": j, "len_diff": ld, "on": a, "off": b})

    metrics_sorted = sorted(metrics, key=lambda x: x["ratio"])
    out_json = os.path.join(base, "injection_text_diff.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    out_txt = os.path.join(base, "injection_text_top_diffs.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for m in metrics_sorted[:3]:
            f.write(f"IDX {m['idx']} ratio={m['ratio']:.3f} jaccard={m['jaccard']:.3f}\nON:\n{m['on']}\nOFF:\n{m['off']}\n{'='*80}\n")

    avg_ratio = sum(m["ratio"] for m in metrics) / len(metrics) if metrics else 0.0
    avg_jaccard = sum(m["jaccard"] for m in metrics) / len(metrics) if metrics else 0.0
    print("avg_ratio", round(avg_ratio, 4))
    print("avg_jaccard", round(avg_jaccard, 4))
    print("n_pairs", len(metrics))
    print("top3_idxs", [m["idx"] for m in metrics_sorted[:3]])


if __name__ == "__main__":
    main()


