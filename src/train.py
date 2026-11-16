"""Train, compare and save the best loan-approval model.
Run:  python -m src.train
"""
import json
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from src.config import *
from src.data_loader import load_clean
from src.preprocessing import FeatureEngineer, build_preprocessor
from src.evaluate import evaluate_model


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(model):
    return Pipeline([("fe", FeatureEngineer()),
                     ("prep", build_preprocessor()),
                     ("clf", model)])


def main():
    df = load_clean()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in get_models().items():
        scores = cross_validate(make_pipeline(model), X_tr, y_tr, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"])
        results[name] = {k: float(np.mean(scores[f"test_{k}"]))
                         for k in ["accuracy", "f1", "roc_auc"]}
        print(name, results[name])

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print("Best model:", best_name)
    best = make_pipeline(get_models()[best_name]).fit(X_tr, y_tr)

    test_metrics = evaluate_model(best, X_te, y_te, FIG_DIR)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(
        {"cv_results": results, "best_model": best_name, "test": test_metrics}, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print("Saved model ->", MODEL_PATH)


if __name__ == "__main__":
    main()
