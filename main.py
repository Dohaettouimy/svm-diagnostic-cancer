"""
=============================================================================
 Apport de la statistique inférentielle dans l'optimisation d'un classifieur
 SVM : Application au diagnostic médical
=============================================================================
 Master : Mathématiques et Ingénierie Numérique
 Faculté des Sciences - Rabat

 Exécution :  python main.py
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import json

from config import ALPHA, RESULTS_DIR, VAR_ETUDE
from data_loader import charger_donnees, nettoyer_donnees, preparer_echantillons
from estimation import (estimation_ponctuelle, intervalle_confiance_proportion,
                        mle_sur_svm)
from exploration import (description_donnees, droite_de_henry, histogramme,
                         tracer_heatmap)
from svm_model import entrainer_svm, evaluer_svm, tracer_frontiere_pca
from tests_hypotheses import (test_conformite, test_homogeneite,
                              test_khi_deux, test_proportion)


def titre(texte):
    print("\n\n" + "#" * 70)
    print(f"#  {texte}")
    print("#" * 70 + "\n")


def main():
    resume = {}

    # ---------------------------------------------------------------
    titre("PARTIE 0 : CHARGEMENT ET NETTOYAGE DES DONNÉES")
    # ---------------------------------------------------------------
    df, _ = charger_donnees()
    df, rapport = nettoyer_donnees(df)
    print("Rapport de nettoyage :")
    for cle, valeur in rapport.items():
        print(f"   {cle:22s} : {valeur}")
    resume["nettoyage"] = rapport

    # ---------------------------------------------------------------
    titre("PARTIE 1 : ANALYSE STATISTIQUE DES DONNÉES")
    # ---------------------------------------------------------------
    description_donnees(df)
    tracer_heatmap(df)
    henry = droite_de_henry(df)
    histogramme(df)
    resume["normalite"] = {
        "shapiro_p_value": float(henry["shapiro"][1]),
        "ks_p_value": float(henry["ks"][1]),
    }

    # ---------------------------------------------------------------
    titre("PARTIE 2 : MODÈLE SVM")
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test, _ = preparer_echantillons(df)
    modele = entrainer_svm(X_train, y_train)
    perf = evaluer_svm(modele, X_test, y_test)
    tracer_frontiere_pca(X_train, y_train)
    resume["svm"] = {"accuracy": float(perf["accuracy"]),
                     "n_test": int(perf["n_test"])}

    # ---------------------------------------------------------------
    titre("PARTIE 3 : ESTIMATION ET INFÉRENCE")
    # ---------------------------------------------------------------
    est = estimation_ponctuelle(df, VAR_ETUDE)
    print()
    ic = intervalle_confiance_proportion(perf["accuracy"], perf["n_test"])
    print()
    mle = mle_sur_svm(X_train, y_train, X_test, y_test)

    resume["estimation"] = {
        "mu_chapeau": float(est["mu_chapeau"]),
        "sigma_e": float(est["sigma_e"]),
        "sigma_chapeau": float(est["sigma_chapeau"]),
    }
    resume["intervalle_confiance"] = {
        "borne_inf": float(ic["ic_inf"]),
        "borne_sup": float(ic["ic_sup"]),
    }
    resume["mle"] = {"a": float(mle["a"]), "b": float(mle["b"]),
                     "log_vraisemblance": float(mle["log_vraisemblance"])}

    # ---------------------------------------------------------------
    titre("PARTIE 4 : VALIDATION STATISTIQUE DU MODÈLE")
    # ---------------------------------------------------------------
    t1 = test_homogeneite(df, VAR_ETUDE)
    print()
    t2 = test_conformite(df, VAR_ETUDE)
    print()
    t3 = test_proportion(perf["accuracy"], perf["n_test"])
    print()
    t4 = test_khi_deux(y_test, perf["y_pred"])

    resume["tests"] = {
        "homogeneite": {"statistique": float(t1["t_obs"]),
                        "p_value": float(t1["p_value"]),
                        "decision": t1["decision"]},
        "conformite": {"statistique": float(t2["t_obs"]),
                       "p_value": float(t2["p_value"]),
                       "decision": t2["decision"]},
        "proportion": {"statistique": float(t3["z_obs"]),
                       "p_value": float(t3["p_value"]),
                       "decision": t3["decision"]},
        "khi_deux": {"statistique": float(t4["chi2_obs"]),
                     "p_value": float(t4["p_value"]),
                     "decision": t4["decision"]},
    }

    # ---------------------------------------------------------------
    titre("CONCLUSION")
    # ---------------------------------------------------------------
    print(f"Précision du SVM                  : {perf['accuracy'] * 100:.2f} %")
    print(f"Intervalle de confiance (95 %)    : "
          f"[{ic['ic_inf'] * 100:.2f} % ; {ic['ic_sup'] * 100:.2f} %]")
    print(f"Variable '{VAR_ETUDE}' discriminante : {t1['decision']}")
    print(f"Conformité à la norme médicale    : {t2['decision']}")
    print(f"Supériorité au standard (p > 0.90): {t3['decision']}")
    print(f"Adéquation du modèle (khi-deux)   : {t4['decision']}")
    print(f"\nSeuil de signification utilisé : alpha = {ALPHA}")
    print("\nLa statistique inférentielle permet de passer de l'analyse d'un")
    print("échantillon à une conclusion valide sur la population, et transforme")
    print("le SVM en un outil de diagnostic validé scientifiquement.")

    # Sauvegarde des résultats
    chemin = RESULTS_DIR / "resultats.json"
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(resume, fichier, indent=2, ensure_ascii=False)
    print(f"\nRésultats enregistrés dans : {chemin}")
    print("Figures enregistrées dans  : figures/")


if __name__ == "__main__":
    main()
