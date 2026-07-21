# Examen DVC et DagsHub

Suivi de données et de modèles avec DVC pour un pipeline de régression prédisant la concentration de silice (`silica_concentrate`) à partir de paramètres opérationnels d'un procédé de flottation de minerai.

## Structure du projet

```
├── data
│   ├── raw_data          # raw.csv (donnée source, trackée par DVC)
│   └── processed_data    # X_train/X_test/y_train/y_test + versions normalisées
├── metrics
│   └── scores.json       # mse, r2 du modèle évalué
├── models
│   ├── best_params.pkl   # meilleurs hyperparamètres (GridSearchCV)
│   └── rf_model.pkl      # RandomForestRegressor entraîné
├── src
│   ├── data
│   │   ├── split_data.py       # split train/test
│   │   └── normalize_data.py   # normalisation (StandardScaler)
│   └── models
│       ├── grid_search.py      # recherche des meilleurs hyperparamètres
│       ├── train_model.py      # entraînement du modèle
│       └── evaluate_model.py   # évaluation + prédictions
├── dvc.yaml               # définition de la pipeline DVC
├── dvc.lock                # état versionné de la pipeline
└── .dvc/config             # configuration du remote DVC (DagsHub)
```

## Installation

```bash
python3 -m venv exam_dvc_env
source exam_dvc_env/bin/activate
pip install pandas scikit-learn numpy click dvc
```

## Reproduire la pipeline

```bash
dvc repro
```

Cela exécute dans l'ordre : `split` -> `normalize` -> `gridsearch` -> `train` -> `evaluate`.

## Récupérer les données et modèles versionnés

```bash
dvc pull
```

## Remote DVC

Les données et modèles sont stockés sur DagsHub : https://dagshub.com/Sebastien6631/examen-dvc

## Auteur

Sebastien Veyssieres — veyssieresebastien66@gmail.com