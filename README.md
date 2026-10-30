# Projet 2 - Détecter des attaques réseau avec le Machine Learning

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py      # entraîne, compare les modèles, écrit models/model.joblib
python src/predict.py    # petit exemple de prédiction
```
Utilisation de la fonction attendue :
```python
from src.predict import predict
predict(df)   # df sans la colonne 'attaque' -> liste de 0/1
```

## Résultats

**Nom / Prénom :** NOM Prénom

Évaluation sur un jeu de validation (25 % des données, stratifié, 750 flux dont 208 attaques). La dernière colonne est le F1 moyen en validation croisée à 5 plis sur les 3000 flux.

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 validation croisée |
|---|---|---|---|---|
| Régression logistique | 0,919 | 0,899 | 0,940 | 0,896 |
| Arbre de décision (profondeur 6) | 0,959 | 0,957 | 0,961 | 0,935 |
| Forêt aléatoire (300 arbres) | 0,973 | 0,971 | 0,976 | 0,960 |

Matrices de confusion sur la validation (VN / FP / FN / VP) : régression logistique 530 / 12 / 21 / 187, arbre 534 / 8 / 9 / 199, forêt 537 / 5 / 6 / 202.

**Modèle retenu et pourquoi :** la forêt aléatoire. Elle a le meilleur F1 de la classe attaque, à la fois sur la validation (0,973) et en validation croisée (0,960), et le meilleur rappel. Les modèles linéaires sont limités ici car les attaques se reconnaissent par des combinaisons de variables (beaucoup de connexions vers la même machine, forte proportion de SYN, échecs d'authentification), que les arbres capturent mieux. Le modèle retenu est ré-entraîné sur les 3000 flux et enregistré dans `models/model.joblib` sous forme de Pipeline (imputation par la médiane, mise à l'échelle, encodage de `protocol`, puis modèle). Les classes sont pondérées (`class_weight="balanced"`) à cause du déséquilibre (≈ 28 % d'attaques).

**Analyse des erreurs (faux négatifs / faux positifs) :**
- **Faux négatifs : 6 attaques ratées sur 208** (rappel 97,1 %). C'est le pire cas en sécurité : une attaque non détectée passe sans alerte et peut aboutir à une intrusion, une machine compromise ou un déni de service, alors qu'une fausse alerte coûte seulement du temps d'analyse. Les attaques ratées ressemblent à du trafic normal : peu de connexions vers la même machine (3 à 9, contre 28 en moyenne pour les attaques), aucun échec d'authentification et un taux de SYN faible.
- **Faux positifs : 5 flux normaux classés en attaque** (précision 97,6 %). Ce sont des flux normaux qui ressemblent à des attaques : par exemple un flux avec 9 échecs d'authentification (mot de passe mal saisi) ou un flux avec 46 connexions vers la même machine et un taux de SYN de 0,90.
- **Pistes d'amélioration :** abaisser le seuil de décision pour augmenter le rappel (au prix de plus de fausses alertes), ajouter des variables dérivées (octets par paquet, ratio des octets envoyés et reçus) et optimiser les hyperparamètres.
