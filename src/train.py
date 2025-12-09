import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_params(path: str = "params.yaml"):
    """Lee parámetros desde el archivo params.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1) Cargar parámetros
    params = load_params()
    data_params = params["data"]
    model_params = params["model"]
    split_params = params["split"]

    # 2) Cargar dataset limpio
    df = pd.read_csv(data_params["processed_path"])

    # Separar features y target
    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3) Definir columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("Columnas numéricas:", num_cols)
    print("Columnas categóricas:", cat_cols)

    # 4) Preprocesamiento: passthrough para numéricas, OneHot para categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # 5) Definir modelo de Regresión Logística
    model = LogisticRegression(
        C=model_params["C"],
        max_iter=model_params["max_iter"],
        n_jobs=-1,
    )

    # Pipeline = preprocesamiento + modelo
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # 6) Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )

    # 7) Entrenar
    pipeline.fit(X_train, y_train)

    # 8) Evaluar
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("Métricas del modelo:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 9) Guardar métricas en JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 10) Guardar modelo entrenado
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model.pkl"
    joblib.dump(pipeline, model_path)

    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
