"""
Module simple pour charger et preparer les images IRM.
Facile a comprendre pour un debutant.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical


ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")
IMAGE_SIZE = 224  # taille standard pour le CNN


def chargement_pretraitement_dataset(chemin_dataset, taille_image=224):
    """
    Parcourt les dossiers de classes (glioma, meningioma, etc.),
    verifie les extensions, gere les images corrompues,
    redimensionne et retourne X, y.

    Exemple:
        X, y = chargement_pretraitement_dataset("data/raw", 224)
        # X shape = (N, 224, 224, 3)  |  y = ["glioma", "notumor", ...]
    """
    if not os.path.exists(chemin_dataset):
        raise FileNotFoundError(f"Dossier introuvable : {chemin_dataset}")

    images = []
    labels = []

    # Chaque sous-dossier = une classe
    classes = sorted(os.listdir(chemin_dataset))
    print(f"Classes detectees : {classes}")

    for nom_classe in classes:
        chemin_classe = os.path.join(chemin_dataset, nom_classe)
        if not os.path.isdir(chemin_classe):
            continue

        print(f" -> Classe : {nom_classe}")

        for nom_fichier in os.listdir(chemin_classe):
            # Verifie l'extension (.jpg, .png, etc.)
            if not nom_fichier.lower().endswith(ALLOWED_EXT):
                print(f"    Ignore (mauvaise extension) : {nom_fichier}")
                continue

            chemin_image = os.path.join(chemin_classe, nom_fichier)

            try:
                img = cv2.imread(chemin_image)
                if img is None:
                    print(f"    Corrompu : {nom_fichier}")
                    continue

                # OpenCV lit en BGR, on convertit en RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Redimensionnement 224x224
                img = cv2.resize(img, (taille_image, taille_image))

                images.append(img)
                labels.append(nom_classe)

            except Exception as e:
                print(f"    Erreur sur {nom_fichier} : {e}")

    if len(images) == 0:
        raise ValueError(f"Aucune image valide dans {chemin_dataset}")

    X = np.array(images)
    y_text = np.array(labels)

    print(f"\nChargement termine : {len(X)} images | {set(y_text)}")
    return X, y_text


def preparer_donnees(X, y_text, test_size=0.2):
   
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_text)
    # Chiffre -> vecteur one-hot ([0,1,0,0])
    y_categorical = to_categorical(y_encoded)

    print("Mapping :", dict(zip(le.classes_, le.transform(le.classes_))))

    # Split 80% train / 20% test, stratify = meme % de chaque classe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded
    )

    # Normalisation : plus stable pour l'entrainement
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Poids equilibres si une classe a moins d'images
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(y_encoded), y=y_encoded
    )
    class_weights = dict(enumerate(class_weights_arr))
    print("Poids des classes :", class_weights)

    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test, le, class_weights
