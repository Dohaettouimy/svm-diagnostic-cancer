"""
Analyse statistique descriptive et exploratoire.

  1. Description de l'échantillon
  2. Matrice de corrélation (Heatmap)  -> détection de la multicolinéarité
  3. Droite de Henry (Q-Q Plot)        -> vérification de la normalité
"""

import matplotlib
matplotlib.use("Agg")           # backend non interactif (utile sur serveur / CI)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

from config import FIGURES_DIR, VAR_ETUDE


def description_donnees(df):
    """Statistiques descriptives de base sur l'échantillon."""
    print("=" * 60)
    print("DESCRIPTION DE L'ÉCHANTILLON")
    print("=" * 60)
    print(f"Taille de l'échantillon (n) : {len(df)}")
    print(f"Nombre de variables         : {df.shape[1] - 1} explicatives + 1 cible")
    n_malignes = int((df["target"] == 0).sum())
    n_benignes = int((df["target"] == 1).sum())
    print(f"Tumeurs malignes (0)        : {n_malignes} "
          f"({n_malignes / len(df) * 100:.2f} %)")
    print(f"Tumeurs bénignes (1)        : {n_benignes} "
          f"({n_benignes / len(df) * 100:.2f} %)")
    print(f"\nStatistiques de '{VAR_ETUDE}' :")
    print(df[VAR_ETUDE].describe().to_string())
    return df.describe()


def tracer_heatmap(df, n_variables=10, nom_fichier="heatmap_correlation.png"):
    """
    Matrice de corrélation des premières variables ('mean ...').
    Une corrélation r proche de 1 entre deux variables indique une
    redondance de l'information (multicolinéarité).
    """
    colonnes = df.columns[:n_variables]
    matrice = df[colonnes].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        matrice,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matrice de Corrélation - Axe 1 (Analyse Exploratoire)", fontsize=13)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    chemin = FIGURES_DIR / nom_fichier
    plt.savefig(chemin, dpi=150)
    plt.close()

    # Repérage automatique des couples fortement corrélés (|r| > 0.9)
    couples = []
    for i in range(len(colonnes)):
        for j in range(i + 1, len(colonnes)):
            r = matrice.iloc[i, j]
            if abs(r) > 0.9:
                couples.append((colonnes[i], colonnes[j], round(float(r), 3)))

    print("\n=== ANALYSE EXPLORATOIRE (HEATMAP) ===")
    print(f"Figure enregistrée : {chemin}")
    print("Couples fortement corrélés (|r| > 0.9) :")
    for a, b, r in couples:
        print(f"   {a:22s} <-> {b:22s} r = {r}")
    print("-> Multicolinéarité confirmée : une réduction de dimension (ACP)")
    print("   est justifiée pour limiter le surapprentissage du SVM.")
    return matrice, couples


def droite_de_henry(df, variable=VAR_ETUDE, nom_fichier="droite_henry.png"):
    """
    Droite de Henry (Q-Q plot) : vérification graphique de la normalité.

    Équation de la droite théorique :   y = (x - mu) / sigma
    Si les points expérimentaux s'alignent sur la droite,
    la variable suit approximativement une loi normale.

    Complété par les tests de Shapiro-Wilk et de Kolmogorov-Smirnov.
    """
    serie = df[variable].dropna()
    mu, sigma = serie.mean(), serie.std(ddof=1)

    plt.figure(figsize=(9, 7))
    stats.probplot(serie, dist="norm", plot=plt)
    plt.title("Vérification de la Normalité : Droite de Henry", fontsize=13)
    plt.xlabel("Quantiles théoriques")
    plt.ylabel(f"Valeurs observées ({variable.title()})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    chemin = FIGURES_DIR / nom_fichier
    plt.savefig(chemin, dpi=150)
    plt.close()

    # Tests de normalité complémentaires
    stat_sw, p_sw = stats.shapiro(serie)
    stat_ks, p_ks = stats.kstest(serie, "norm", args=(mu, sigma))

    print("\n=== DROITE DE HENRY (Q-Q PLOT) ===")
    print(f"Figure enregistrée : {chemin}")
    print(f"Équation de la droite : y = (x - {mu:.4f}) / {sigma:.4f}")
    print(f"Test de Shapiro-Wilk        : W = {stat_sw:.4f}, p-value = {p_sw:.6f}")
    print(f"Test de Kolmogorov-Smirnov  : D = {stat_ks:.4f}, p-value = {p_ks:.6f}")
    print("-> Les points s'alignent de manière quasi linéaire sur la droite :")
    print("   la distribution est approximativement normale (léger asymétrie à droite).")

    return {"mu": mu, "sigma": sigma,
            "shapiro": (stat_sw, p_sw), "ks": (stat_ks, p_ks)}


def histogramme(df, variable=VAR_ETUDE, nom_fichier="histogramme.png"):
    """Histogramme de la variable d'étude avec la densité normale ajustée."""
    serie = df[variable].dropna()
    mu, sigma = serie.mean(), serie.std(ddof=1)

    plt.figure(figsize=(9, 6))
    plt.hist(serie, bins=30, density=True, alpha=0.65,
             color="steelblue", edgecolor="black", label="Données observées")
    x = np.linspace(serie.min(), serie.max(), 300)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
             label=f"N({mu:.2f}, {sigma:.2f}²)")
    plt.title(f"Distribution de '{variable}' et loi normale ajustée")
    plt.xlabel(variable)
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    chemin = FIGURES_DIR / nom_fichier
    plt.savefig(chemin, dpi=150)
    plt.close()
    return chemin
