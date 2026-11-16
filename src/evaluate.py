"""Metrics + plots (confusion matrix, ROC, feature importance)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)


def evaluate_model(pipe, X_te, y_te, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }
    metrics = {k: float(v) for k, v in metrics.items()}
    print("Test metrics:", metrics)

    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=["Rejected", "Approved"])
    plt.title("Confusion Matrix"); plt.savefig(fig_dir / "confusion_matrix.png", dpi=150); plt.close()

    RocCurveDisplay.from_predictions(y_te, proba)
    plt.title("ROC Curve"); plt.savefig(fig_dir / "roc_curve.png", dpi=150); plt.close()

    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = getattr(clf, "feature_importances_", None)
    if imp is None and hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    if imp is not None:
        idx = np.argsort(imp)[-10:]
        plt.figure(figsize=(7, 4))
        plt.barh([names[i].split("__")[-1] for i in idx], imp[idx])
        plt.title("Top 10 Feature Importances"); plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance.png", dpi=150); plt.close()
    return metrics
