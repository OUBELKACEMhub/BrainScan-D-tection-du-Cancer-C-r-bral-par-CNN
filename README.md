# 🧠 BrainScan AI - Classification des Tumeurs Cérébrales par CNN
Application intelligente capable de classer les IRM en 4 classes : `glioma`, `meningioma`, `notumor`, `pituitary`.

## 📁 Structure du Dataset
```
Dataset1/
└── Data/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

```

## ⚙️ Installation
```bash
pip install opencv-python tensorflow scikit-learn matplotlib seaborn streamlit numpy
```

## 🚀 Utilisation
```python
from main import *

X, y_text = chargement_pretraitement_dataset("Dataset1/Data", 224)
visualiser_dataset(X, y_text)
X_train, X_test, y_train, y_test, le, class_weights = preparer_donnees(X, y_text)
model = conception_modele_cnn((224,224,3), 4)
model = compiler_modele(model)
history = entrainer_modele(model, X_train, y_train, X_test, y_test, class_weights)
evaluer_modele(model, history, X_test, y_test, le)
predict_image("test.jpg", model, le)
```
```bash
streamlit run app.py
```

---

## 📖 Documentation Technique - Choix des Méthodes par Fonction

### PARTIE 1 : Chargement et Prétraitement

#### 1. `chargement_pretraitement_dataset(chemin_dataset, taille_image=224)`
**Rôle :** Charger les images depuis le disque et faire le prétraitement de base.

**Méthodes utilisées :**
- `os.listdir()` pour parcourir les 4 dossiers (chaque dossier = une classe)
- Vérification d'extension `('.jpg','.jpeg','.png','.bmp')`
- Bloc `try-except` pour chaque image
- `cv2.imread()` + `cv2.cvtColor(BGR->RGB)` + `cv2.resize(224,224)`

**Choix techniques :**
| Choix | Justification |
|---|---|
| Vérification d'extension | Supprime les fichiers corrompus qui font planter l'entraînement. Exigé par le cahier des charges. |
| `try-except` | Empêche l'arrêt total du programme si une image est illisible. |
| `OpenCV (cv2)` | Beaucoup plus rapide que PIL pour le `resize`, travaille directement en tableau NumPy. |
| Taille `224x224` | Standard ImageNet, compatible avec Transfer Learning (ResNet, EfficientNet). Compromis idéal entre détails de la tumeur et coût mémoire. |
| Conversion `BGR->RGB` | `cv2` lit en BGR alors que `matplotlib` et `TensorFlow` attendent du RGB. |

#### 2. `visualiser_dataset(X, y_text)`
**Rôle :** Explorer visuellement le dataset pour le rapport.

**Méthodes utilisées :**
- `sns.countplot()` pour la distribution des classes
- `plt.subplots()` pour afficher 1 échantillon par classe

**Choix techniques :**
- `seaborn` plus visuel qu'un simple `print` pour détecter le déséquilibre des classes.
- Affichage d'un échantillon par classe pour vérifier la qualité des IRM.

#### 3. `preparer_donnees(X, y_text, test_size=0.2)`
**Rôle :** Rendre les données exploitables par le CNN.

**Méthodes utilisées :**
- `LabelEncoder()` : texte -> chiffre (`glioma->0`)
- `to_categorical()` : chiffre -> vecteur One-Hot (`0->[1,0,0,0]`)
- `train_test_split(stratify=y_categorical)` : 80% train / 20% test
- Normalisation `X / 255.0` -> plage `[0,1]`
- `compute_class_weight('balanced')`

**Choix techniques :**
| Choix | Justification |
|---|---|
| `LabelEncoder + to_categorical` | Le CNN ne comprend que les chiffres. One-Hot obligatoire pour `categorical_crossentropy`. |
| `stratify` | **Très important.** Garde le même % de chaque classe dans train et test. Sans ça, le test est biaisé. |
| Normalisation `[0,1]` | Pixels 0-255 = gradients instables. Avec `/255.0`, convergence 2x plus rapide et +10% accuracy. |
| `class_weight='balanced'` | Donne un poids plus grand aux classes minoritaires pour ne pas toujours prédire `notumor`. |

### PARTIE 2 : Conception du Modèle

#### 4. `conception_modele_cnn(input_shape, nb_classes=4)`
**Rôle :** Construire l'architecture CNN.

**Architecture :**
```
Conv2D(32) -> BatchNorm -> MaxPooling -> Dropout(0.25)
Conv2D(64) -> BatchNorm -> MaxPooling -> Dropout(0.25)
Conv2D(128) -> BatchNorm -> MaxPooling -> Dropout(0.3)
Conv2D(256) -> BatchNorm -> MaxPooling -> Dropout(0.3)
GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.5) -> Dense(4, softmax)
```

**Choix techniques :**
| Choix | Justification |
|---|---|
| Progression `32->256` filtres | Les premières couches détectent les contours, les dernières les formes complexes de la tumeur. |
| Kernel `(3,3)` | Le plus efficace, standard VGG/ResNet. Peu de paramètres pour un grand champ réceptif. |
| Activation `ReLU` | `max(0,x)`, rapide et évite le Vanishing Gradient de `sigmoid`. |
| `BatchNormalization` | Accélère l'entraînement x2 et stabilise le modèle. |
| `MaxPooling2D(2,2)` | Divise la taille par 2, -75% de paramètres, rend invariant à la position de la tumeur. |
| `Dropout` | Évite le surapprentissage (Overfitting) sur petit dataset médical. |
| `GlobalAveragePooling2D` | **Choix stratégique.** Remplace `Flatten` (50k params) par une moyenne (256 params). -90% de paramètres, modèle plus léger. |

#### 5. `compiler_modele(model)`
**Rôle :** Compiler le modèle.

**Méthodes utilisées :**
- `model.compile(optimizer='adam', loss='categorical_crossentropy')`
- `model.summary()` et `plot_model()`

**Choix techniques :**
| Choix | Justification |
|---|---|
| Optimiseur `Adam` | Combine Momentum + RMSProp, adapte le LR automatiquement. Plus stable que SGD. |
| Loss `categorical_crossentropy` | Standard pour classification multi-classe avec Softmax. |
| Activation `Softmax` | Donne des probabilités dont la somme = 1. |
| `summary() / plot_model()` | Exigés par le cahier des charges pour vérifier l'architecture. |

### PARTIE 3 : Entraînement et Évaluation

#### 6. `entrainer_modele(model, X_train, y_train, ..., epochs=30, batch_size=32)`
**Rôle :** Lancer l'apprentissage.

**Méthodes utilisées :**
- `ModelCheckpoint(save_best_only=True, monitor='val_accuracy')`
- `EarlyStopping(patience=5, monitor='val_loss')`
- `time.time()` pour la durée
- `model.fit(class_weight=...)`

**Choix techniques :**
| Choix | Justification |
|---|---|
| `batch_size=32` | Compromis idéal entre stabilité du gradient et mémoire GPU. |
| `EarlyStopping` | Arrête si `val_loss` ne s'améliore plus pendant 5 epochs. Évite l'overfitting. |
| `ModelCheckpoint` | Sauvegarde uniquement le meilleur modèle dans `best_brainscan_model.h5`. |
| `time.time()` | Mesure la durée exacte demandée par le cahier des charges. |

#### 7. `evaluer_modele(model, history, X_test, y_test, le)`
**Rôle :** Évaluer sur des données jamais vues.

**Méthodes utilisées :**
- `model.evaluate()` pour l'accuracy test
- Courbes `Accuracy/Loss` avec `matplotlib`
- `confusion_matrix` + `seaborn.heatmap`
- `classification_report` (Precision/Recall/F1)

**Choix techniques :**
- Courbes d'apprentissage pour détecter l'Overfitting visuellement.
- Matrice de confusion indispensable en médecine pour voir si `glioma` est confondu avec `meningioma`.

#### 8. `predict_image(path, model, le)`
**Rôle :** Prédire une seule image IRM (déploiement).

**Méthodes utilisées :**
- `cv2.imread -> cvtColor -> resize -> /255.0 -> np.expand_dims -> model.predict -> argmax`

**Choix techniques :**
- Refaire **exactement** le même prétraitement que l'entraînement est obligatoire sinon prédiction fausse.
- `expand_dims` ajoute la dimension batch `(1,224,224,3)` attendue par le modèle.

## 📊 Hyperparamètres
| Paramètre | Valeur |
|---|---|
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 30 avec EarlyStopping |
| Optimiseur | Adam |
| Loss | categorical_crossentropy |

## 👨‍💻 Auteur
Projet BrainScan AI - 2025
```