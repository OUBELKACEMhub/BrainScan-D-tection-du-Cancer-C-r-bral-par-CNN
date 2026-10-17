# BrainScan AI - Classification Tumeurs Cerebrales (CNN)

Projet simple pour debutant : classifier des IRM en 4 classes : `glioma`, `meningioma`, `notumor`, `pituitary`.

## Structure (simple)

```
brainscan-ai/
  app/app.py                    -> Interface Streamlit simple
  src/
    data_preprocessing.py       -> Charger + preparer les images
    model_architecture.py       -> Creer le CNN
    train.py                    -> Entrainer
    evaluate.py                 -> Evaluer
    predict.py                  -> Predire une image
  notebooks/
    01_data_loading_preprocessing.ipynb  -> Tester chargement
    02_eda_visualization.ipynb           -> Voir le dataset
    03_model_training.ipynb              -> Entrainer le modele
    04_evaluation.ipynb                  -> Tester accuracy + matrice
    05_optimization.ipynb                -> Chercher meilleurs hyperparams
  data/raw/                     -> 4 dossiers : glioma, meningioma, notumor, pituitary
  models/                       -> best_model.keras apres entrainement
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

**1. Tester les fonctions (notebooks)** - Ouvre 01 -> 05 dans l'ordre :
- 01 teste `chargement_pretraitement_dataset` + `preparer_donnees`
- 02 visualisation
- 03 `conception_modele_cnn` + `entrainer_modele`
- 04 `evaluer_modele`
- 05 Grid Search pour optimiser

**2. Lancer l'app**
```bash
streamlit run app/app.py
# ou avec Docker :
docker compose up --build
# -> http://localhost:8501
```

**3. En Python (1 ligne par etape)**
```python
from src.data_preprocessing import chargement_pretraitement_dataset, preparer_donnees
from src.model_architecture import conception_modele_cnn, compiler_modele
from src.train import entrainer_modele
from src.evaluate import evaluer_modele

X, y = chargement_pretraitement_dataset("data/raw", 224)
X_train, X_test, y_train, y_test, le, class_weights = preparer_donnees(X, y)
model = conception_modele_cnn((224,224,3), 4)
model = compiler_modele(model)
history = entrainer_modele(model, X_train, y_train, X_test, y_test, class_weights, epochs=30)
evaluer_modele(model, history, X_test, y_test, le)
```

## Comment ca marche ? (tres simple)

1. **chargement_pretraitement_dataset** - parcourt les 4 dossiers, verifie extensions, try/except, BGR->RGB, resize 224.
2. **preparer_donnees** - LabelEncoder + one-hot, split 80/20 stratify, /255, class_weights.
3. **conception_modele_cnn** - 4 blocs Conv(32->256)+BatchNorm+Pool+Dropout -> GAP->Dense->Softmax.
4. **compiler_modele** - Adam lr=0.0001 + categorical_crossentropy.
5. **entrainer_modele** - augmentation + Checkpoint + EarlyStopping(patience 5).
6. **evaluer_modele** - accuracy, courbes, rapport, matrice confusion.
7. **predict_image** - meme pretraitement + predict + argmax.

Hyperparametres : LR 0.0001 | Batch 32 | Epochs 30 + EarlyStopping | Adam

Auteur : Projet BrainScan AI - 2025
