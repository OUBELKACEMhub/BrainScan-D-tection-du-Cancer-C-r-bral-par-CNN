"""
Entrainement du modele avec augmentation et callbacks.
"""

import time
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def entrainer_modele(model, X_train, y_train, X_test, y_test, class_weights, epochs=30, batch_size=32):
    """
    Entraine le modele :
    - Augmentation sur le train (rotation, zoom, flip) pour eviter l'overfitting
    - Sauvegarde le meilleur modele
    - Arrete si val_loss ne s'ameliore plus (EarlyStopping)

    Retourne history (pour afficher les courbes apres)
    """
    # Augmentation : cree des variantes des images d'entrainement
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest"
    )

    # Callbacks
    Path("models").mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        "models/best_model.keras",   # meilleur modele
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,                  # arrete si pas d'amelioration pendant 5 epochs
        restore_best_weights=True,
        verbose=1
    )

    start = time.time()

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print(f"Duree d'entrainement : {(time.time() - start) / 60:.2f} min")

    # Sauvegarde le modele final
    model.save("models/final_model.keras")
    print("Modele sauvegarde : models/final_model.keras")
    print("Meilleur modele : models/best_model.keras")

    return history

