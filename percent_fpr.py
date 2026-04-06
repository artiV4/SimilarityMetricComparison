import sys
import math
from ks_eval.data import load_dsl_strong_password_csv
from ks_eval.preprocess import PreprocessConfig
import ks_eval.analysis_threshold
from ks_eval.threshold_cache import load_threshold, upsert_threshold

if len(sys.argv) < 3:
    print(
        "Usage: python percent_fpr.py <metric> <fpr_threshold> "
        "[power=none|yeo-johnson] [scaler=none|standard|robust] "
    "[winsor=low,high] [minvar=float] [clip=float] [rank=int]"
    )
    sys.exit(1)

metric = sys.argv[1].strip().lower()
fpr_threshold_percent = float(sys.argv[2])

# Optional args: key=value
opts = {}
for arg in sys.argv[3:]:
    if "=" not in arg:
        continue
    k, v = arg.split("=", 1)
    opts[k.strip().lower()] = v.strip()

csv_path = "./DSL-StrongPasswordData.csv"

power = opts.get("power", "none")
scaler = opts.get("scaler", "standard")
clip = float(opts["clip"]) if "clip" in opts else 5.0


winsor = None
if "winsor" in opts:
    lo_s, hi_s = opts["winsor"].split(",")
    winsor = (float(lo_s), float(hi_s))

minvar = float(opts["minvar"]) if "minvar" in opts else None

recompute = opts.get("recompute", "false").strip().lower() in {"1", "true", "yes", "y"}
thresholds_csv = opts.get("thresholds", "thresholds.csv")

preprocess = PreprocessConfig(
    impute="median",
    power_transform=power,
    scaler=scaler,
    winsorize=winsor,
    min_variance=minvar,
    clip=clip,
)

print_rank = ''
if "rank" in opts and metric.startswith("mahalanobis"):
    # Use supported syntax in make_scorer(): mahalanobis:rank=<int>
    metric = f"{metric.split(':', 1)[0]}:rank={int(opts['rank'])}"
    print_rank = f"rank={opts['rank']}"


print(f"\nSimilarity Metric - {metric} {print_rank}")
print(f"FPR Threshold - {fpr_threshold_percent}")

dataset = load_dsl_strong_password_csv(csv_path)

baseline_fraction = 0.10
min_baseline = 1
impostor_per_subject = 200
seed = 42

cached_t = None
if not recompute:
    cached_t = load_threshold(
        csv_path=thresholds_csv,
        metric=metric,
        fpr=fpr_threshold_percent,
        baseline_fraction=baseline_fraction,
        min_baseline=min_baseline,
        impostor_per_subject=impostor_per_subject,
        seed=seed,
        preprocess=preprocess,
    )

if cached_t is None:
    genuine, impostor = ks_eval.analysis_threshold.compute_trials_by_name(
        dataset,
        metric=metric,
        preprocess=preprocess,
        baseline_fraction=baseline_fraction,
        min_baseline=min_baseline,
        impostor_per_subject=impostor_per_subject,
        random_state=seed,
    )

    k = int(math.floor(fpr_threshold_percent * len(impostor)))

    # Decide threshold direction based on metric type.
    # - distances (euclidean/mahalanobis): accept if score <= t
    # - similarities (cosine/gaussian log-likelihood): accept if score >= t
    scorer = ks_eval.analysis_threshold.make_scorer(metric)
    accept_greater = bool(getattr(scorer, "greater_is_better", False))

    res = ks_eval.analysis_threshold.threshold_at_most_k_fp(
        genuine_distances=genuine,
        impostor_distances=impostor,
        k=k,
        accept_greater=accept_greater,
    )

    upsert_threshold(
        csv_path=thresholds_csv,
        metric=metric,
        fpr=fpr_threshold_percent,
        baseline_fraction=baseline_fraction,
        min_baseline=min_baseline,
        impostor_per_subject=impostor_per_subject,
        seed=seed,
        preprocess=preprocess,
        threshold=res.threshold,
    )
else:
    # We still need trials to report TPR/FPR; but we can avoid recomputing the
    # threshold search. (Kept simple for now.)
    genuine, impostor = ks_eval.analysis_threshold.compute_trials_by_name(
        dataset,
        metric=metric,
        preprocess=preprocess,
        baseline_fraction=baseline_fraction,
        min_baseline=min_baseline,
        impostor_per_subject=impostor_per_subject,
        random_state=seed,
    )
    res_t = float(cached_t)
    scorer = ks_eval.analysis_threshold.make_scorer(metric)
    accept_greater = bool(getattr(scorer, "greater_is_better", False))
    if accept_greater:
        fp = int((impostor >= res_t).sum())
        tp = int((genuine >= res_t).sum())
        fn = int((genuine < res_t).sum())
    else:
        fp = int((impostor <= res_t).sum())
        tp = int((genuine <= res_t).sum())
        fn = int((genuine > res_t).sum())

    class _Res:
        threshold = res_t
        false_positives = fp
        true_positives = tp
        false_negatives = fn

    res = _Res()

    k = int(math.floor(fpr_threshold_percent * len(impostor)))

print(f"impostor_trials={len(impostor)} => target_k=floor({fpr_threshold_percent*100}%)={k}")
print(f"t={res.threshold}")
print(f"false_positives={res.false_positives} ({res.false_positives/len(impostor):.4%})")
print(f"true_positives={res.true_positives} ({res.true_positives/len(genuine):.4%})")
print(f"false_negatives={res.false_negatives} ({res.false_negatives/len(genuine):.4%})")