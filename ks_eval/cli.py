from __future__ import annotations

import argparse
from pathlib import Path

from ks_eval.data import load_dsl_strong_password_csv
from ks_eval.evaluator import evaluate_sequential_probes
from ks_eval.preprocess import PreprocessConfig
from ks_eval.scoring import (
    CosineSimilarityScorer,
    EuclideanDistanceScorer,
    GaussianLogLikelihoodScorer,
    MahalanobisDistanceScorer,
)
from ks_eval.analysis_threshold import compute_trials_by_name, threshold_at_most_k_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Keystroke similarity evaluation framework")
    parser.add_argument("--csv", type=str, required=True, help="Path to DSL-StrongPasswordData.csv")
    parser.add_argument(
        "--baseline-fraction",
        type=float,
        default=0.10,
        help="Fraction of each subject's samples used for baseline enrollment (default: 0.10)",
    )
    parser.add_argument(
        "--min-baseline",
        type=int,
        default=1,
        help="Minimum baseline samples per subject (default: 1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output path (.parquet or .csv). If omitted, prints head().",
    )

    # --- Preprocessing switches ---
    parser.add_argument(
        "--impute",
        choices=["none", "mean", "median"],
        default="none",
        help="Missing value imputation fitted on baseline (default: none)",
    )
    parser.add_argument(
        "--scaler",
        choices=["none", "standard", "robust"],
        default="none",
        help="Feature scaling fitted on baseline (default: none)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="Optional clipping after scaling, e.g. 5.0 => clip to [-5,5]",
    )
    parser.add_argument(
        "--abs",
        dest="abs_values",
        action="store_true",
        help="Take absolute value of features (rarely needed)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print basic dataset and probe summary.",
    )

    # --- Threshold utility (euclidean) ---
    parser.add_argument(
        "--euclidean-threshold-k",
        type=int,
        default=None,
        help=(
            "If set, computes an acceptance threshold t (accept if distance<=t) "
            "for Euclidean distance with at most K false positives, prints it, and exits."
        ),
    )

    parser.add_argument(
        "--gaussian-threshold-k",
        type=int,
        default=None,
        help=(
            "If set, computes an acceptance threshold t (accept if gaussian_ll>=t) "
            "with at most K false positives, prints it, and exits."
        ),
    )
    parser.add_argument(
        "--accept-greater",
        action="store_true",
        help=(
            "Accept if score >= threshold (useful for similarity/likelihood metrics). "
            "If not set, accept if score <= threshold (distance metrics)."
        ),
    )
    parser.add_argument(
        "--impostor-per-subject",
        type=int,
        default=200,
        help="Impostor probes sampled per subject when estimating threshold (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for impostor sampling (default: 42)",
    )

    # --- Scoring modules (optional) ---
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Comma-separated metrics to compute: euclidean,cosine,mahalanobis,gaussian_ll. "
            "If omitted, no scores are computed (probes only)."
        ),
    )

    args = parser.parse_args()

    dataset = load_dsl_strong_password_csv(args.csv)

    preprocess = PreprocessConfig(
        impute=args.impute,
        scaler=args.scaler,
        clip=args.clip,
        abs_values=args.abs_values,
    )

    if args.euclidean_threshold_k is not None:
        # Use euclidean distance and derive threshold with <=K false positives.
        # Tip: use --impute median --scaler standard for covariance-free Euclidean robustness.
        genuine, impostor = compute_trials_by_name(
            dataset,
            metric="euclidean",
            preprocess=preprocess,
            baseline_fraction=args.baseline_fraction,
            min_baseline=args.min_baseline,
            impostor_per_subject=args.impostor_per_subject,
            random_state=args.seed,
        )
        res = threshold_at_most_k_fp(
            genuine_distances=genuine,
            impostor_distances=impostor,
            k=int(args.euclidean_threshold_k),
            accept_greater=bool(args.accept_greater),
        )
        print(
            "Euclidean threshold (accept if distance <= t)\n"
            f"t = {res.threshold}\n"
            f"false_positives = {res.false_positives} / {res.n_impostor_trials}\n"
            f"true_positives  = {res.true_positives} / {res.n_genuine_trials}\n"
            f"false_negatives = {res.false_negatives} / {res.n_genuine_trials}\n"
        )
        return

    if args.gaussian_threshold_k is not None:
        genuine, impostor = compute_trials_by_name(
            dataset,
            metric="gaussian",
            preprocess=preprocess,
            baseline_fraction=args.baseline_fraction,
            min_baseline=args.min_baseline,
            impostor_per_subject=args.impostor_per_subject,
            random_state=args.seed,
        )
        res = threshold_at_most_k_fp(
            genuine_distances=genuine,
            impostor_distances=impostor,
            k=int(args.gaussian_threshold_k),
            accept_greater=True,
        )
        print(
            "Gaussian log-likelihood threshold (accept if score >= t)\n"
            f"t = {res.threshold}\n"
            f"false_positives = {res.false_positives} / {res.n_impostor_trials}\n"
            f"true_positives  = {res.true_positives} / {res.n_genuine_trials}\n"
            f"false_negatives = {res.false_negatives} / {res.n_genuine_trials}\n"
        )
        return

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    scorers = []
    for m in metric_list:
        if m == "euclidean":
            scorers.append(EuclideanDistanceScorer())
        elif m == "cosine":
            scorers.append(CosineSimilarityScorer())
        elif m == "mahalanobis":
            scorers.append(MahalanobisDistanceScorer())
        elif m in {"gaussian", "gaussian_ll"}:
            scorers.append(GaussianLogLikelihoodScorer())
        else:
            raise ValueError(f"Unknown metric: {m}")

    probe_df = evaluate_sequential_probes(
        dataset,
        baseline_fraction=args.baseline_fraction,
        min_baseline=args.min_baseline,
        preprocess=preprocess,
        scorers=scorers,
        score_fns={},
    )

    if args.summary:
        df = dataset.df
        print(f"Loaded: {len(df)} samples, {df['subject'].nunique()} subjects")
        print(f"Features: {len(dataset.feature_columns)}")
        print(f"Generated: {len(probe_df)} probe rows")

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".parquet":
            probe_df.to_parquet(out_path, index=False)
        elif out_path.suffix.lower() == ".csv":
            probe_df.to_csv(out_path, index=False)
        else:
            raise ValueError("--out must end with .parquet or .csv")
    else:
        print(probe_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
