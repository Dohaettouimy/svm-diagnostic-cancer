# Apport de la statistique inférentielle dans l'optimisation d'un classifieur SVM

**Application au diagnostic médical (cancer du sein)**

Projet de Statistique Inférentielle Avancée — Master Mathématiques et Ingénierie Numérique, Faculté des Sciences de Rabat.

---

## Problématique

Comment l'inférence statistique peut-elle transformer le SVM d'un simple algorithme de calcul en un outil de diagnostic médical fiable et validé ?

L'objectif est de classifier des tumeurs en **malignes** ou **bénignes** à l'aide d'un SVM, puis d'utiliser la statistique inférentielle pour valider scientifiquement les résultats et les généraliser de l'échantillon à la population.

## Données

| | |
|---|---|
| Source | Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository |
| Accès | `sklearn.datasets.load_breast_cancer` (aucun téléchargement nécessaire) |
| Taille de l'échantillon | 569 observations (patients) |
| Variables explicatives | 30 variables quantitatives (rayon, texture, périmètre, aire, etc.) |
| Variable cible | Diagnostic qualitatif : 0 = maligne, 1 = bénigne |
| Échantillonnage | Aléatoire simple, stratifié sur la cible (80 % train / 20 % test) |

## Structure du dépôt

```
svm-inference-statistique/
├── main.py                   # Pipeline complet : exécute toute l'étude
├── requirements.txt
├── README.md
├── src/
│   ├── config.py             # Paramètres centralisés (alpha, mu_0, p_0, seed...)
│   ├── data_loader.py        # Chargement, nettoyage, échantillonnage, standardisation
│   ├── exploration.py        # Description, heatmap, droite de Henry, histogramme
│   ├── svm_model.py          # Entraînement, évaluation, frontière de décision (ACP)
│   ├── estimation.py         # Estimation ponctuelle, intervalle de confiance, MLE
│   └── tests_hypotheses.py   # Student, Z-test sur proportion, khi-deux
├── notebooks/
│   └── projet_statistique.ipynb   # Version notebook, cellule par cellule
├── figures/                  # Graphiques générés automatiquement
└── results/                  # resultats.json
```

## Installation et exécution

```bash
git clone https://github.com/<votre-compte>/svm-inference-statistique.git
cd svm-inference-statistique

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

Le script affiche l'intégralité de l'étude dans le terminal, enregistre les figures dans `figures/` et les résultats numériques dans `results/resultats.json`.

## Méthodologie

### 1. Analyse statistique des données

- **Nettoyage** : suppression des doublons, contrôle des valeurs manquantes, vérification des types.
- **Heatmap de corrélation** : met en évidence une corrélation quasi parfaite (r ≈ 1.00) entre `mean radius`, `mean perimeter` et `mean area`. Cette **multicolinéarité** justifie une réduction de dimension (ACP) pour limiter le surapprentissage.
- **Droite de Henry (Q-Q plot)** : vérification graphique de la normalité de `mean radius`, d'équation

  $$y = \frac{1}{\sigma}(x - \mu)$$

  complétée par les tests de Shapiro-Wilk et de Kolmogorov-Smirnov.

### 2. Modèle SVM

Le SVM cherche un hyperplan optimal $w \cdot x + b = 0$ maximisant la marge $2/\lVert w \rVert$ :

$$\min_{w,b} \tfrac{1}{2}\lVert w \rVert^2 \quad \text{sous la contrainte} \quad y_i(w \cdot x_i + b) \ge 1$$

La décision se fait par $y_{pred} = \mathrm{signe}(w \cdot x + b)$.

### 3. Estimation et inférence

- **Estimation ponctuelle** : $\hat{\mu} = \bar{x}$ (estimateur sans biais) et $\hat{\sigma} = \sigma_e \sqrt{n/(n-1)}$.
- **Intervalle de confiance** de la précision du SVM (TCL applicable car $n \ge 30$) :

  $$IC = \left[f - t_\alpha\sqrt{\frac{f(1-f)}{n}} \; ; \; f + t_\alpha\sqrt{\frac{f(1-f)}{n}}\right]$$

- **Maximum de vraisemblance (MLE)** : les distances géométriques à l'hyperplan sont transformées en probabilités par la sigmoïde $\sigma(x) = 1/(1 + e^{-(ax+b)})$, dont les paramètres $a$ et $b$ sont estimés en maximisant la log-vraisemblance. L'implémentation manuelle (optimisation BFGS) est comparée à la calibration de Platt de scikit-learn.

### 4. Validation statistique du modèle

| Test | Hypothèses | Objectif |
|---|---|---|
| Student d'homogénéité | H0 : μ_M = μ_B | La variable est-elle discriminante ? |
| Student de conformité | H0 : μ = 14.0 | L'échantillon est-il conforme à la norme médicale ? |
| Z-test sur une proportion | H0 : p = 0.90 contre H1 : p > 0.90 | Le modèle dépasse-t-il le standard médical ? |
| Khi-deux d'adéquation | H0 : modèle adéquat | Les effectifs prédits suivent-ils la distribution réelle ? |

## Résultats obtenus

Avec `RANDOM_STATE = 42` (voir `src/config.py`) :

| Indicateur | Valeur |
|---|---|
| Moyenne estimée `mean radius` | 14.13 mm |
| Écart-type estimé (sans biais) | 3.52 mm |
| Précision du SVM (échantillon test, n = 114) | 97.37 % |
| Intervalle de confiance à 95 % | [94.43 % ; 100 %] |
| Paramètres MLE de la sigmoïde | a ≈ 2.19 ; b ≈ -0.14 |

| Test | Statistique | p-value | Décision |
|---|---|---|---|
| Student d'homogénéité | t = 22.21 | 1.7 × 10⁻⁶⁴ | Rejet de H0 — variable fortement discriminante |
| Student de conformité | t = 0.86 | 0.3893 | Non-rejet de H0 — échantillon conforme |
| Z-test sur proportion | z = 2.62 | 0.0044 | Rejet de H0 — modèle supérieur au standard |
| Khi-deux d'adéquation | χ² = 0.0377 | 0.8460 | Non-rejet de H0 — bonne adéquation |

Les résultats varient légèrement selon la graine aléatoire et la taille du test ; modifier `RANDOM_STATE` ou `TEST_SIZE` dans `src/config.py` permet de reproduire d'autres découpages.

## Conclusion

La performance d'un SVM en diagnostic médical doit être validée par la statistique inférentielle. Le maximum de vraisemblance fournit une interprétation probabiliste des sorties du classifieur, tandis que les tests de Student, le Z-test sur une proportion et le test du khi-deux confirment la significativité de la variable discriminante et la fiabilité du modèle. Le SVM constitue ainsi un outil de diagnostic robuste et scientifiquement validé.

## Dépendances

numpy, pandas, scipy, scikit-learn, matplotlib, seaborn (voir `requirements.txt`).

## Auteur

**Doha Et-touimy** — encadrée par M. Mourad El Azhari.
