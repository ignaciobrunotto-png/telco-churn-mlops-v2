import os
from pathlib import Path

import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    """Carga parámetros globales desde params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza específica para el dataset de churn de TelcoVision."""

    # 1) Eliminar identificador que no aporta al modelo
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # 2) Asegurar tipos numéricos (por si en otra versión vienen como string)
    numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "churn"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Eliminar filas con nulos (en este dataset no hay, pero es defensivo)
    df = df.dropna()

    # 4) Asegurar que churn sea entero 0/1
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)

    return df


def main():
    params = load_params()
    raw_path = Path(params["data"]["raw_path"])
    processed_path = Path(params["data"]["processed_path"])

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar dataset crudo
    df = pd.read_csv(raw_path)

    # Aplicar limpieza específica
    df_clean = clean_telco(df)

    # Guardar dataset limpio
    df_clean.to_csv(processed_path, index=False)
    print(f"Dataset limpio guardado en: {processed_path}")
    print(f"Shape final: {df_clean.shape}")


if __name__ == "__main__":
    main()
