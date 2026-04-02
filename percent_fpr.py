import math
from ks_eval.data import load_dsl_strong_password_csv
from ks_eval.preprocess import PreprocessConfig
import ks_eval.analysis_threshold

metric = "cosine"


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

k = int(math.floor(0.01 * len(impostor)))
res = ks_eval.analysis_threshold.threshold_at_most_k_fp(genuine_distances=genuine, impostor_distances=impostor, k=k)
print(f"impostor_trials={len(impostor)} => target_k=floor(1%)={k}")
print(f"t={res.threshold}")
print(f"false_positives={res.false_positives} ({res.false_positives/len(impostor):.4%})")
print(f"true_positives={res.true_positives} ({res.true_positives/len(genuine):.4%})")
print(f"false_negatives={res.false_negatives} ({res.false_negatives/len(genuine):.4%})")