import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


def compute_metrics(labels, probs, threshold=0.5, clip_duration_sec=2.0):
    """
    Parameters
    ----------
    labels            : list or array of true labels (0=normal, 1=infiltration)
    probs             : list or array of positive-class probabilities
    threshold         : risk score threshold for triggering an alarm
    clip_duration_sec : duration of one clip in seconds (T_LEN / fps)
                        default = 16 frames / 8 fps = 2 seconds

    Returns
    -------
    dict with keys: auc, aucpr, sensitivity, specificity,
                    fa_per_hour, threshold
    """
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)

    # ── primary: AUC ──────────────────────────────────────────────────
    auc = roc_auc_score(labels, probs)

    # ── secondary: AUCPR ──────────────────────────────────────────────
    aucpr = average_precision_score(labels, probs)

    # ── confusion matrix components ───────────────────────────────────
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    # ── sensitivity = TP / (TP + FN) ─────────────────────────────────
    sensitivity = tp / max(tp + fn, 1e-6)

    # ── specificity = TN / (TN + FP) ─────────────────────────────────
    specificity = tn / max(tn + fp, 1e-6)

    # ── FA/h — false alarms per hour of monitoring ────────────────────
    # Each negative clip that fires = one false alarm.
    # Total monitoring time = all clips × clip duration.
    n_false_alarms = fp
    total_hours = (len(labels) * clip_duration_sec) / 3600
    fa_per_hour = n_false_alarms / max(total_hours, 1e-6)

    return {
        "auc": round(auc, 4),
        "aucpr": round(aucpr, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "fa_per_hour": round(fa_per_hour, 4),
        "threshold": threshold,
        # raw counts — useful for debugging
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def find_best_threshold(labels, probs, target_fa_h=0.3,
                        clip_duration_sec=2.0):
    """
    Sweep ROC thresholds and return the highest threshold that keeps
    FA/h at or below target_fa_h.

    Use this at evaluation time to tune the alarm rule rather than
    using the default 0.5 cutoff.

    Parameters
    ----------
    labels            : true labels
    probs             : positive-class probabilities
    target_fa_h       : maximum acceptable false alarms per hour
                        (proposal target: 0.3)
    clip_duration_sec : seconds per clip

    Returns
    -------
    float — best threshold value
    """
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    n_negatives = np.sum(labels == 0)
    total_hours = (len(labels) * clip_duration_sec) / 3600

    best_threshold = 0.5  # fallback if no threshold meets the target
    for thr, fp_rate in zip(thresholds, fpr):
        n_fp = fp_rate * n_negatives
        fa_h = n_fp / max(total_hours, 1e-6)
        if fa_h <= target_fa_h:
            best_threshold = float(thr)

    return best_threshold


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict returned by compute_metrics."""
    print("-" * 38)
    print(f"  AUC          : {metrics['auc']:.4f}   (target ≥ 0.85)")
    print(f"  AUCPR        : {metrics['aucpr']:.4f}")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"  Specificity  : {metrics['specificity']:.4f}")
    print(f"  FA/h         : {metrics['fa_per_hour']:.4f}  (target ≤ 0.30)")
    print(f"  Threshold    : {metrics['threshold']:.4f}")
    print(f"  TP/TN/FP/FN  : {metrics['tp']} / {metrics['tn']} "
          f"/ {metrics['fp']} / {metrics['fn']}")
    print("-" * 38)
