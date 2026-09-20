"""
Validation statistique du modèle : tests d'hypothèses.

  1. Test de Student d'homogénéité   (comparaison de deux moyennes)
  2. Test de Student de conformité   (comparaison à une norme)
  3. Test d'hypothèse sur une proportion (Z-test unilatéral)
  4. Test du Khi-deux d'adéquation   (ajustement du modèle aux données)
"""

import numpy as np
from scipy.stats import chi2, norm, ttest_1samp, ttest_ind

from config import (ALPHA, CHI2_CRITIQUE, MU_0, P_0, VAR_ETUDE,
                    Z_ALPHA, Z_ALPHA_2)


# ---------------------------------------------------------------------------
# 1. TEST DE STUDENT D'HOMOGÉNÉITÉ
# ---------------------------------------------------------------------------
def test_homogeneite(df, variable=VAR_ETUDE, alpha=ALPHA):
    """
    Objectif : déterminer si la variable permet de distinguer
               les classes Maligne (M) et Bénigne (B).

        H0 : mu_M = mu_B   (la variable n'est pas discriminante)
        H1 : mu_M != mu_B  (la variable est significative)

        t_obs = (x_A - x_B) / sqrt(s_A²/n_A + s_B²/n_B)

    Règle de décision : |t_obs| > t_alpha/2  =>  rejet de H0
    """
    groupe_M = df[df["target"] == 0][variable]   # tumeurs malignes
    groupe_B = df[df["target"] == 1][variable]   # tumeurs bénignes

    # equal_var=False -> test de Welch (variances non supposées égales)
    t_obs, p_value = ttest_ind(groupe_M, groupe_B, equal_var=False)

    print("=" * 60)
    print("TEST 1 : TEST DE STUDENT D'HOMOGÉNÉITÉ")
    print("=" * 60)
    print(f"Variable testée      : {variable}")
    print(f"H0 : mu_M = mu_B     |  H1 : mu_M != mu_B")
    print(f"Moyenne Maligne      : {groupe_M.mean():.4f}  (n = {len(groupe_M)})")
    print(f"Moyenne Bénigne      : {groupe_B.mean():.4f}  (n = {len(groupe_B)})")
    print(f"Statistique t_obs    : {t_obs:.4f}")
    print(f"Valeur critique      : t_alpha/2 = {Z_ALPHA_2}")
    print(f"p-value              : {p_value:.3e}")

    if p_value < alpha:
        print(f"-> Décision   : Rejet de H0 (p < {alpha})")
        print("-> Conclusion : Le rayon moyen est une variable SIGNIFICATIVE "
              "(fortement discriminante).")
        decision = "Rejet de H0"
    else:
        print(f"-> Décision   : Non-rejet de H0 (p >= {alpha})")
        print("-> Conclusion : Le rayon moyen n'est pas une variable significative.")
        decision = "Non-rejet de H0"

    return {"t_obs": t_obs, "p_value": p_value, "decision": decision}


# ---------------------------------------------------------------------------
# 2. TEST DE STUDENT DE CONFORMITÉ
# ---------------------------------------------------------------------------
def test_conformite(df, variable=VAR_ETUDE, mu_0=MU_0, alpha=ALPHA):
    """
    Objectif : vérifier si la moyenne de l'échantillon est conforme
               à la norme médicale mu_0.

        H0 : mu = mu_0   (échantillon conforme)
        H1 : mu != mu_0  (écart significatif)

        z_obs = (x_barre - mu_0) / (s / sqrt(n))

    Règle de décision : |z_obs| > z_alpha/2  =>  rejet de H0
    """
    serie = df[variable].dropna()
    t_obs, p_value = ttest_1samp(serie, popmean=mu_0)

    print("=" * 60)
    print("TEST 2 : TEST DE STUDENT DE CONFORMITÉ")
    print("=" * 60)
    print(f"Norme de référence (mu_0)  : {mu_0}")
    print(f"H0 : mu = {mu_0}          |  H1 : mu != {mu_0}")
    print(f"Moyenne de l'échantillon   : {serie.mean():.4f}")
    print(f"Statistique t_obs          : {t_obs:.4f}")
    print(f"Valeur critique            : z_alpha/2 = {Z_ALPHA_2}")
    print(f"p-value                    : {p_value:.4f}")

    if p_value < alpha:
        print(f"-> Décision   : Rejet de H0 (p < {alpha})")
        print("-> Conclusion : L'écart est significatif par rapport à la norme.")
        decision = "Rejet de H0"
    else:
        print(f"-> Décision   : Non-rejet de H0 (p >= {alpha})")
        print("-> Conclusion : L'échantillon est conforme à la norme médicale.")
        decision = "Non-rejet de H0"

    return {"t_obs": t_obs, "p_value": p_value, "decision": decision}


# ---------------------------------------------------------------------------
# 3. TEST D'HYPOTHÈSE SUR UNE PROPORTION
# ---------------------------------------------------------------------------
def test_proportion(f, n, p_0=P_0, alpha=ALPHA):
    """
    Objectif : vérifier si la précision du SVM dépasse le standard médical.

        H0 : p = p_0   (performance conforme au standard)
        H1 : p > p_0   (performance supérieure)  -> test unilatéral à droite

        z_obs = (f - p_0) / sqrt(p_0 (1 - p_0) / n)

    Règle de décision : z_obs > z_alpha  =>  rejet de H0
    """
    z_obs = (f - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
    p_value = 1 - norm.cdf(z_obs)

    print("=" * 60)
    print("TEST 3 : TEST D'HYPOTHÈSE SUR UNE PROPORTION (Z-TEST)")
    print("=" * 60)
    print(f"Standard médical (p_0)  : {p_0}")
    print(f"H0 : p = {p_0}         |  H1 : p > {p_0}")
    print(f"Précision observée (f)  : {f:.4f}   (n = {n})")
    print(f"Statistique z_obs       : {z_obs:.4f}")
    print(f"Valeur critique         : z_alpha = {Z_ALPHA}")
    print(f"p-value                 : {p_value:.4f}")

    if z_obs > Z_ALPHA:
        print(f"-> Décision   : Rejet de H0 (z_obs > {Z_ALPHA})")
        print("-> Conclusion : Le modèle est SUPÉRIEUR au standard médical.")
        decision = "Rejet de H0"
    else:
        print(f"-> Décision   : Non-rejet de H0 (z_obs <= {Z_ALPHA})")
        print("-> Conclusion : La performance du modèle est conforme au standard, "
              "sans supériorité démontrée.")
        decision = "Non-rejet de H0"

    return {"z_obs": z_obs, "p_value": p_value, "decision": decision}


# ---------------------------------------------------------------------------
# 4. TEST DU KHI-DEUX D'ADÉQUATION
# ---------------------------------------------------------------------------
def test_khi_deux(y_test, y_pred, alpha=ALPHA):
    """
    Objectif : comparer la distribution des prédictions du SVM
               avec la distribution réelle des classes.

        H0 : le modèle est adéquat   (pas de différence significative)
        H1 : le modèle est inadéquat (différence significative)

        khi2_obs = somme (O_i - T_i)² / T_i

    Règle de décision : khi2_obs > khi2_(1-alpha)(nu)  =>  rejet de H0
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_test, y_pred]))

    effectifs_reels = np.array([(y_test == c).sum() for c in classes], dtype=float)
    effectifs_predits = np.array([(y_pred == c).sum() for c in classes], dtype=float)

    chi2_obs = float(np.sum((effectifs_predits - effectifs_reels) ** 2
                            / effectifs_reels))
    ddl = len(classes) - 1                       # degrés de liberté
    valeur_critique = chi2.ppf(1 - alpha, ddl)
    p_value = 1 - chi2.cdf(chi2_obs, ddl)

    print("=" * 60)
    print("TEST 4 : TEST DU KHI-DEUX D'ADÉQUATION")
    print("=" * 60)
    print(f"Effectifs réels (T_i)    : {effectifs_reels.astype(int).tolist()}")
    print(f"Effectifs prédits (O_i)  : {effectifs_predits.astype(int).tolist()}")
    print(f"Degrés de liberté        : {ddl}")
    print(f"Statistique khi2_obs     : {chi2_obs:.4f}")
    print(f"Valeur critique          : {valeur_critique:.2f} "
          f"(valeur tabulée : {CHI2_CRITIQUE})")
    print(f"p-value                  : {p_value:.4f}")

    if chi2_obs > valeur_critique:
        print(f"-> Décision   : Rejet de H0 ({chi2_obs:.3f} > {valeur_critique:.2f})")
        print("-> Conclusion : Le modèle est INADÉQUAT (différence significative).")
        decision = "Rejet de H0"
    else:
        print(f"-> Décision   : Non-rejet de H0 ({chi2_obs:.3f} <= {valeur_critique:.2f})")
        print("-> Conclusion : Le modèle est ADÉQUAT (bonne adéquation).")
        decision = "Non-rejet de H0"

    return {"chi2_obs": chi2_obs, "p_value": p_value,
            "ddl": ddl, "decision": decision}
