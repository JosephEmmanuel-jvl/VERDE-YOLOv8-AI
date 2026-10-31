"""Build comparison.json and the metric tables in reports/METRIC_TABLES.md from eval/*. Run from workspace root."""
import json, os

T = ("baseline_640", "candidate_640", "baseline_416", "candidate_416")
M = {t: json.load(open(f"eval/{t}/metrics.json"))["per_class"] for t in T}
F = {t: json.load(open(f"eval/{t}/fixed_threshold_metrics.json")) for t in T}
CL = ("overall", "Healthy", "Yellowing", "Wilting")
KEYS = ("precision", "recall", "f1", "ap50", "ap50_95")
pc = lambda v: round(v * 100, 2)
PAIRS = {"640_vs_640 (frozen template size)": ("baseline_640", "candidate_640"), "416_vs_416 (baseline's trained size)": ("baseline_416", "candidate_416"),
         "native: baseline@416 vs candidate@640": ("baseline_416", "candidate_640")}
out = {"metrics": M, "fixed_conf_0_25_iou_0_5": F, "deltas_candidate_minus_baseline_points": {}}
md = ["# Metric tables (historical 200-image test)\n", "All values are percentages. Primary P/R/F1 use Ultralytics' best mean-F1 operating point; `@.25` columns use conf>=0.25, class-aware greedy IoU>=0.5.\n"]
for name, (b, c) in PAIRS.items():
    md += [f"\n## {name}\n", "| Class | Model | P | R | F1 | AP50 | AP50-95 | P@.25 | R@.25 | F1@.25 | TP/FP/FN@.25 |", "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    out["deltas_candidate_minus_baseline_points"][name] = {}
    for k in CL:
        for t, lab in ((b, "baseline"), (c, "candidate")):
            f = F[t][k]
            md.append(f"| {k} | {lab} ({t.split('_')[1]}) | " + " | ".join(f"{pc(M[t][k][m]):.2f}" for m in KEYS) + f" | {pc(f['precision']):.2f} | {pc(f['recall']):.2f} | {pc(f['f1']):.2f} | {f['tp']}/{f['fp']}/{f['fn']} |")
        d = {m: round(pc(M[c][k][m]) - pc(M[b][k][m]), 2) for m in KEYS}
        d.update({m + "@.25": round(pc(F[c][k][m]) - pc(F[b][k][m]), 2) for m in ("precision", "recall", "f1")})
        out["deltas_candidate_minus_baseline_points"][name][k] = d
        md.append(f"| {k} | **delta** | " + " | ".join(f"{d[m]:+.2f}" for m in KEYS) + " | " + " | ".join(f"{d[m + '@.25']:+.2f}" for m in ("precision", "recall", "f1")) + " | |")
os.makedirs("reports", exist_ok=True)
json.dump(out, open("reports/comparison.json", "w"), indent=1)
open("reports/METRIC_TABLES.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("\n".join(md))
