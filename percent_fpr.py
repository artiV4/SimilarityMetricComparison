import sys
import math
from ks_eval.data import load_dsl_strong_password_csv
from ks_eval.preprocess import PreprocessConfig
import ks_eval.analysis_threshold

if(len(sys.argv) != 3):
    print("Usage: python percent_fpr.py <metric> <fpr_threshold>")
    sys.exit(1)

metric = sys.argv[1].strip().lower()
fpr_threshold_percent = float(sys.argv[2])

print(f"Similarity Metric - {metric}")
print(f"FPR Threshold - {fpr_threshold_percent}")

csv_path = "./DSL-StrongPasswordData.csv"
preprocess = PreprocessConfig(impute="median", scaler="standard", clip=5.0)

dataset = load_dsl_strong_password_csv(csv_path)
genuine, impostor = ks_eval.analysis_threshold.compute_trials_by_name(
    dataset,
    metric=metric,
    preprocess=preprocess,
    baseline_fraction=0.10,
    min_baseline=1,
    impostor_per_subject=200,
    random_state=42,
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
print(f"impostor_trials={len(impostor)} => target_k=floor({fpr_threshold_percent*100}%)={k}")
print(f"t={res.threshold}")
print(f"false_positives={res.false_positives} ({res.false_positives/len(impostor):.4%})")
print(f"true_positives={res.true_positives} ({res.true_positives/len(genuine):.4%})")
print(f"false_negatives={res.false_negatives} ({res.false_negatives/len(genuine):.4%})")