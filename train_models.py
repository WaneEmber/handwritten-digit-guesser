import json
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

models = {
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf", C=5, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ))
    ])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro")),
        "recall_macro": float(recall_score(y_test, preds, average="macro")),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        )
    }
    joblib.dump(model, f"{name}_model.joblib")
    print(f"\n{name.upper()} RESULTS")
    print(f"Accuracy: {results[name]['accuracy']:.4%}")
    print(f"Macro Precision: {results[name]['precision_macro']:.4%}")
    print(f"Macro Recall: {results[name]['recall_macro']:.4%}")
    print(f"Macro F1: {results[name]['f1_macro']:.4%}")
    print(classification_report(y_test, preds, zero_division=0))

report = {
    "dataset": "Scikit-learn Digits",
    "samples": int(len(X)),
    "image_size": "8x8",
    "features": 64,
    "split": "80% training / 20% testing (stratified)",
    "random_state": RANDOM_STATE,
    "results": results
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nSaved knn_model.joblib, svm_model.joblib, metrics.json")
