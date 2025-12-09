import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml"):
    """Lee parámetros desde params.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1) Parámetros
    params = load_params()
    data_params = params["data"]
    split_params = params["split"]

    processed_path = data_params["processed_path"]
    model_path = Path("models/model.pkl")

    # 2) Cargar dataset limpio y modelo
    df = pd.read_csv(processed_path)
    model = joblib.load(model_path)

    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3) Mismo train/test split que en train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )

    # 4) Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5) Directorio de reportes
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # 6) ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Telco Churn")
    plt.legend(loc="lower right")
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # 7) Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix - Telco Churn")
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # 8) Classification report (texto)
    cls_report = classification_report(y_test, y_pred)
    report_path = reports_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(cls_report)

    # 9) Guardar también un JSON con las métricas clave
    metrics_eval = {
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }
    with open(reports_dir / "evaluation_summary.json", "w") as f:
        json.dump(metrics_eval, f, indent=4)

    print(f"ROC curve guardada en: {roc_path}")
    print(f"Matriz de confusión guardada en: {cm_path}")
    print(f"Classification report en: {report_path}")


if __name__ == "__main__":
    main()
