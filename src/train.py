"""Projet 2 - Detection d'attaques reseau avec le Machine Learning.

Lancer depuis la racine du depot : python src/train.py
- explore les donnees
- compare plusieurs modeles sur un jeu de validation
- reentraine le meilleur sur toutes les donnees
- enregistre le Pipeline dans models/model.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data" / "flux_train.csv"
MODELE = RACINE / "models" / "model.joblib"
SEED = 42

COLONNES_NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
                "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
COLONNES_CAT = ["protocol"]


def explorer(df):
    print("=== Exploration ===")
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Repartition de 'attaque' :\n", df["attaque"].value_counts(normalize=True).round(3))
    print()


def construire_pipeline(modele):
    """Pretraitement + modele dans un seul Pipeline."""
    pretraitement = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), COLONNES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CAT),
    ])
    return Pipeline([("prep", pretraitement), ("modele", modele)])


def main():
    df = pd.read_csv(DATA)
    explorer(df)

    X = df[COLONNES_NUM + COLONNES_CAT]
    y = df["attaque"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    candidats = {
        "Regression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de decision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Foret aleatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultats = []
    for nom, modele in candidats.items():
        pipe = construire_pipeline(modele).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        f1_cv = cross_val_score(construire_pipeline(modele), X, y, cv=cv, scoring="f1").mean()
        resultats.append({
            "Modele": nom,
            "F1 (attaque)": f1_score(y_val, pred),
            "Rappel (attaque)": recall_score(y_val, pred),
            "Precision (attaque)": precision_score(y_val, pred),
            "F1 validation croisee (5 plis)": f1_cv,
        })
        print(f"=== {nom} ===")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"]))
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"Matrice de confusion : VN={tn} FP={fp} FN={fn} VP={tp}\n")
        resultats[-1].update({"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)})

    tableau = pd.DataFrame(resultats).set_index("Modele")
    print("=== Comparaison ===")
    print(tableau.round(3).to_string())

    # Choix du meilleur : F1 moyen en validation croisee (plus stable qu'un seul decoupage)
    meilleur = tableau["F1 validation croisee (5 plis)"].idxmax()
    print(f"\nModele retenu : {meilleur}")

    # Reentrainement sur TOUTES les donnees et sauvegarde
    final = construire_pipeline(candidats[meilleur]).fit(X, y)
    MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODELE)
    print("Modele enregistre dans", MODELE)


if __name__ == "__main__":
    main()
