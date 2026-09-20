"""
Chargement, nettoyage et échantillonnage des données.

Source : Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
         disponible directement via sklearn.datasets.load_breast_cancer
n = 569 observations, 30 variables explicatives quantitatives,
1 variable cible qualitative (0 = Maligne, 1 = Bénigne).
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE


def charger_donnees():
    """Charge le jeu de données et retourne un DataFrame avec la cible."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target          # 0 = Maligne, 1 = Bénigne
    return df, data


def nettoyer_donnees(df):
    """
    Nettoyage du jeu de données :
      - suppression des doublons
      - vérification et traitement des valeurs manquantes (médiane)
      - vérification des types
    Retourne le DataFrame nettoyé et un rapport de nettoyage.
    """
    rapport = {
        "lignes_initiales": len(df),
        "valeurs_manquantes": int(df.isnull().sum().sum()),
        "doublons": int(df.duplicated().sum()),
    }

    # Suppression des doublons éventuels
    df = df.drop_duplicates().copy()

    # Imputation par la médiane si des valeurs manquantes existent
    colonnes_num = df.select_dtypes(include="number").columns
    if df[colonnes_num].isnull().sum().sum() > 0:
        df[colonnes_num] = df[colonnes_num].fillna(df[colonnes_num].median())

    rapport["lignes_finales"] = len(df)
    return df, rapport


def preparer_echantillons(df):
    """
    Échantillonnage aléatoire simple (EAS) : découpage train / test
    avec stratification sur la variable cible pour conserver les proportions.
    Les variables explicatives sont centrées-réduites (standardisation),
    étape indispensable pour le SVM qui est sensible à l'échelle.
    """
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df, data = charger_donnees()
    df, rapport = nettoyer_donnees(df)
    print("=== NETTOYAGE DES DONNÉES ===")
    for cle, valeur in rapport.items():
        print(f"{cle:25s} : {valeur}")
    print(f"\nTaille du jeu de données : {df.shape}")
    print(f"Répartition des classes  :\n{df['target'].value_counts()}")
    print(f"\n{df.head()}")
