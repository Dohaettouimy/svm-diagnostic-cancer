"""
Estimation et inférence statistique.

  1. Estimation ponctuelle des paramètres (mu, sigma)
  2. Intervalle de confiance de la performance du SVM
  3. Maximum de Vraisemblance (MLE) appliqué aux sorties du SVM
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.svm import SVC

from config import RANDOM_STATE, SVM_C, SVM_KERNEL, VAR_ETUDE, Z_ALPHA_2


# ---------------------------------------------------------------------------
# 1. ESTIMATION PONCTUELLE
# ---------------------------------------------------------------------------
def estimation_ponctuelle(df, variable=VAR_ETUDE):
    """
    La moyenne empirique est un estimateur sans biais de mu :
        mu_chapeau = x_barre

    L'écart-type empirique (ddof=0) est biaisé ; on le corrige :
        sigma_chapeau = sigma_e * sqrt(n / (n - 1))
    ce qui équivaut exactement à l'écart-type calculé avec ddof=1.
    """
    serie = df[variable].dropna()
    n = len(serie)

    m_e = serie.mean()                    # moyenne observée
    sigma_e = serie.std(ddof=0)           # écart-type observé (biaisé)
    sigma_chapeau = sigma_e * np.sqrt(n / (n - 1))   # estimateur sans biais
    sigma_ddof1 = serie.std(ddof=1)       # vérification

    print("=" * 60)
    print("ESTIMATION PONCTUELLE DES PARAMÈTRES (mu, sigma)")
    print("=" * 60)
    print(f"Taille de l'échantillon n        : {n}")
    print(f"Moyenne observée (mu_chapeau)    : {m_e:.4f} mm")
    print(f"Écart-type observé (sigma_e)     : {sigma_e:.4f} mm")
    print(f"Écart-type estimé sans biais     : {sigma_chapeau:.4f} mm")
    print(f"Vérification avec ddof=1         : {sigma_ddof1:.4f} mm")

    return {"n": n, "mu_chapeau": m_e, "sigma_e": sigma_e,
            "sigma_chapeau": sigma_chapeau}


# ---------------------------------------------------------------------------
# 2. INTERVALLE DE CONFIANCE DE LA PERFORMANCE DU SVM
# ---------------------------------------------------------------------------
def intervalle_confiance_proportion(f, n, t_alpha=Z_ALPHA_2, niveau=95):
    """
    Intervalle de confiance d'une proportion (précision du SVM).

    Conditions de validité : n >= 30 -> Théorème Central Limite applicable.

        IC = [ f - t_alpha * sqrt(f(1-f)/n) ; f + t_alpha * sqrt(f(1-f)/n) ]
    """
    erreur_standard = np.sqrt(f * (1 - f) / n)
    ic_inf = f - t_alpha * erreur_standard
    ic_sup = f + t_alpha * erreur_standard

    # La précision est une proportion : on borne l'intervalle dans [0, 1]
    ic_inf, ic_sup = max(0.0, ic_inf), min(1.0, ic_sup)

    print("=" * 60)
    print(f"INTERVALLE DE CONFIANCE À {niveau} % DE LA PERFORMANCE DU SVM")
    print("=" * 60)
    print(f"Condition de validité  : n = {n} >= 30  -> TCL applicable")
    print(f"Précision observée (f) : {f * 100:.2f} %")
    print(f"Erreur standard        : {erreur_standard:.6f}")
    print(f"Valeur critique        : t_alpha = {t_alpha}")
    print(f"Intervalle de confiance ({niveau} %) : "
          f"[{ic_inf * 100:.2f} % ; {ic_sup * 100:.2f} %]")

    return {"f": f, "n": n, "erreur_standard": erreur_standard,
            "ic_inf": ic_inf, "ic_sup": ic_sup}


# ---------------------------------------------------------------------------
# 3. MAXIMUM DE VRAISEMBLANCE (MLE)
# ---------------------------------------------------------------------------
def _log_vraisemblance_negative(params, distances, y):
    """
    Opposé de la log-vraisemblance du modèle sigmoïde de Platt.

        sigma(x) = 1 / (1 + exp(-(a*x + b)))

        log L(a, b) = somme [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]

    On minimise -log L, ce qui revient à maximiser la vraisemblance.
    """
    a, b = params
    z = a * distances + b
    # log(1 + exp(-z)) calculé de façon numériquement stable
    log_p = -np.logaddexp(0, -z)          # log(sigma(z))
    log_1_moins_p = -np.logaddexp(0, z)   # log(1 - sigma(z))
    return -np.sum(y * log_p + (1 - y) * log_1_moins_p)


def mle_platt_manuel(distances, y):
    """
    Estimation des paramètres (a, b) de la sigmoïde par maximum
    de vraisemblance, résolue numériquement (méthode BFGS).
    """
    resultat = minimize(
        _log_vraisemblance_negative,
        x0=np.array([1.0, 0.0]),
        args=(np.asarray(distances), np.asarray(y)),
        method="BFGS",
    )
    a, b = resultat.x
    return a, b, -resultat.fun     # a, b et la log-vraisemblance maximale


def mle_sur_svm(X_train, y_train, X_test, y_test, n_affiches=5):
    """
    Transformation des sorties géométriques du SVM en probabilités
    à l'aide de la fonction sigmoïde, dont les paramètres sont estimés
    par la méthode du Maximum de Vraisemblance (calibration de Platt).

    Deux approches sont comparées :
      - implémentation manuelle (optimisation de la log-vraisemblance)
      - implémentation de scikit-learn (SVC(probability=True))
    """
    # --- Distances géométriques à l'hyperplan (sorties brutes du SVM) ---
    modele = SVC(kernel=SVM_KERNEL, C=SVM_C, random_state=RANDOM_STATE)
    modele.fit(X_train, y_train)
    distances_train = modele.decision_function(X_train)
    distances_test = modele.decision_function(X_test)

    # --- MLE manuel ---
    a, b, log_vrais = mle_platt_manuel(distances_train, np.asarray(y_train))

    # --- MLE via scikit-learn (Platt scaling intégré) ---
    modele_proba = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True,
                       random_state=RANDOM_STATE)
    modele_proba.fit(X_train, y_train)
    probs_sklearn = modele_proba.predict_proba(X_test)

    # Probabilités issues de notre sigmoïde estimée par MLE
    p_benigne = 1 / (1 + np.exp(-(a * distances_test + b)))
    p_maligne = 1 - p_benigne

    print("=" * 60)
    print("MAXIMUM DE VRAISEMBLANCE (MLE) APPLIQUÉ AU SVM")
    print("=" * 60)
    print("Modèle : sigma(x) = 1 / (1 + exp(-(a*x + b)))")
    print(f"Paramètres estimés par MLE : a = {a:.4f} , b = {b:.4f}")
    print(f"Log-vraisemblance maximale : {log_vrais:.4f}")

    resultats = pd.DataFrame({
        "Distance (Géométrique)": distances_test[:n_affiches],
        "Probabilité Maligne (via MLE)": p_maligne[:n_affiches],
        "Probabilité Bénigne (via MLE)": p_benigne[:n_affiches],
        "P. Bénigne (sklearn)": probs_sklearn[:n_affiches, 1],
    })
    print(f"\n{resultats.to_string(float_format=lambda v: f'{v:.6f}')}")

    return {"a": a, "b": b, "log_vraisemblance": log_vrais,
            "distances_test": distances_test,
            "p_maligne": p_maligne, "p_benigne": p_benigne,
            "tableau": resultats}
