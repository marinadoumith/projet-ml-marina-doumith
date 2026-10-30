"""Prediction avec le Pipeline enregistre dans models/model.joblib."""
from pathlib import Path

import joblib
import pandas as pd

MODELE = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_modele = None


def predict(df: pd.DataFrame) -> list:
    """df : memes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    global _modele
    if _modele is None:
        _modele = joblib.load(MODELE)
    X = df.drop(columns=["attaque"], errors="ignore")
    return [int(p) for p in _modele.predict(X)]


if __name__ == "__main__":
    donnees = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "flux_train.csv")
    print(predict(donnees.head(10)))
