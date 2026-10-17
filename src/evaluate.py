"""
Evaluation du modele : accuracy, courbes, matrice de confusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluer_modele(model, history, X_test, y_test, le):
    """
    Evalue le modele et affiche :
    - Accuracy sur le test
    - Courbes Accuracy / Loss
    - Rapport de classification (precision, recall, f1)
    - Matrice de confusion

    Exemple:
        evaluer_modele(model, history, X_test, y_test, le)
    """
    # Accuracy test
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy Test : {acc*100:.2f}%")

    # Courbes d'apprentissage
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Rapport
    print("\n--- Rapport de classification ---")
    print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predit")
    plt.ylabel("Reel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "loss": loss, "confusion_matrix": cm}

