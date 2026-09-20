"""
Configuration globale du projet.
Tous les paramètres modifiables sont centralisés ici.
"""

from pathlib import Path

# ---------- Chemins ----------
ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Reproductibilité ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 114 observations de test sur 569

# ---------- Variable d'étude ----------
VAR_ETUDE = "mean radius"

# ---------- Paramètres statistiques ----------
ALPHA = 0.05              # seuil de signification
MU_0 = 14.0               # norme médicale (test de conformité)
P_0 = 0.90                # standard médical (test sur une proportion)

# Valeurs critiques (loi normale centrée réduite)
Z_ALPHA_2 = 1.96          # bilatéral, alpha = 0.05
Z_ALPHA = 1.645           # unilatéral, alpha = 0.05
CHI2_CRITIQUE = 3.84      # khi-deux, 1 degré de liberté, alpha = 0.05

# ---------- Modèle SVM ----------
SVM_KERNEL = "linear"
SVM_C = 1.0
