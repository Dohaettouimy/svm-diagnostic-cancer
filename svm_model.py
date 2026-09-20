"""
Modèle SVM (Support Vector Machine) et visualisation de la frontière de décision.

Rappel théorique
----------------
Hyperplan séparateur       :  w . x + b = 0
Fonction de décision       :  y_pred = signe(w . x + b)
Marge                      :  2 / ||w||
Problème d'optimisation    :  min (1/2)||w||^2  sous  y_i (w . x_i + b) >= 1
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.svm import SVC

from config import FIGURES_DIR, RANDOM_STATE, SVM_C, SVM_KERNEL


def entrainer_svm(X_train, y_train, kernel=SVM_KERNEL, C=SVM_C):
    """Entraîne un classifieur SVM sur les données d'entraînement."""
    modele = SVC(kernel=kernel, C=C, random_state=RANDOM_STATE)
    modele.fit(X_train, y_train)
    return modele


def evaluer_svm(modele, X_test, y_test):
    """Évalue le modèle : précision, matrice de confusion, rapport détaillé."""
    y_pred = modele.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    mc = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("PERFORMANCE DU CLASSIFIEUR SVM")
    print("=" * 60)
    print(f"Noyau utilisé            : {modele.kernel}")
    print(f"Vecteurs de support      : {int(modele.n_support_.sum())}")
    print(f"Taille du test (n)       : {len(y_test)}")
    print(f"Précision (accuracy) f   : {precision * 100:.2f} %")
    print(f"\nMatrice de confusion :\n{mc}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Maligne', 'Bénigne'])}")

    return {"y_pred": y_pred, "accuracy": precision,
            "matrice_confusion": mc, "n_test": len(y_test)}


def tracer_frontiere_pca(X_train, y_train, nom_fichier="frontiere_svm_pca.png"):
    """
    Projection des données sur les 2 premières composantes principales (ACP)
    puis tracé de la frontière de décision du SVM dans ce plan 2D.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_train)

    modele_2d = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RANDOM_STATE)
    modele_2d.fit(X_2d, y_train)

    # Grille de points pour colorier les régions de décision
    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = modele_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, alpha=0.25,
                 cmap=ListedColormap(["#5566cc", "#e8a090"]))
    for classe, couleur, libelle in [(0, "royalblue", "Maligne (0)"),
                                     (1, "crimson", "Bénigne (1)")]:
        masque = np.asarray(y_train) == classe
        plt.scatter(X_2d[masque, 0], X_2d[masque, 1],
                    c=couleur, edgecolor="k", s=42, label=libelle)

    variance = pca.explained_variance_ratio_
    plt.title("Frontière de Décision du Classifieur SVM (Visualisation 2D via PCA)")
    plt.xlabel(f"Composante Principale 1 (PCA1) - {variance[0]*100:.1f} % var.")
    plt.ylabel(f"Composante Principale 2 (PCA2) - {variance[1]*100:.1f} % var.")
    plt.legend(loc="upper left")
    plt.tight_layout()
    chemin = FIGURES_DIR / nom_fichier
    plt.savefig(chemin, dpi=150)
    plt.close()

    print(f"\nFigure de la frontière de décision enregistrée : {chemin}")
    print(f"Variance expliquée par les 2 composantes : "
          f"{variance.sum() * 100:.2f} %")
    return pca, modele_2d
