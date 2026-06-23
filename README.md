# XAI-MedVision — Détection de nodules pulmonaires explicable

Détection de nodules pulmonaires sur images CT par **apprentissage auto-supervisé
(I-JEPA)** suivi d'une **sonde linéaire**, avec des explications visuelles
(**Grad-CAM, LIME, SHAP**) pour rendre chaque prédiction interprétable.

Le projet couvre toute la chaîne : EDA → prétraitement → pré-entraînement SSL →
sonde linéaire → comparaison XAI → évaluation, plus une **interface web de démo**
(FastAPI + React) qui accepte des images 2D et des volumes 3D.

---

## Démarrage rapide (démo web)

Prérequis : **Python 3.11** (PyTorch ne supporte pas encore 3.14) et **Node.js 18+**.

```bash
# 1. Cloner le projet
git clone https://github.com/Zainabfarih/XAI-MedVision
cd XAI-MedVision

# 2. Télécharger les modèles depuis Hugging Face (active le mode "live")
pip install huggingface_hub
python scripts/download_models.py
```

```bash
# 3. Backend (terminal 1)
cd ui/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

```bash
# 4. Frontend (terminal 2)
cd ui/frontend
npm install
npm run dev
```

Ouvrir **http://localhost:3000**, puis charger une image 2D ou un volume 3D
depuis [`data/samples/`](data/samples/). Sans l'étape 2, la démo tourne en
**mode démo** (résultats simulés) ; avec les modèles, en **mode live** (vraies
prédictions). Détails : [`ui/README.md`](ui/README.md).

---

## Démo en vidéo

<!-- Glissez votre vidéo ici depuis l'éditeur web GitHub, ou remplacez par votre lien. -->

> _Vidéo de démonstration à venir._

---

## Résultats

Encodeur ViT-Small/16 pré-entraîné avec I-JEPA (auto-supervisé, sans labels),
puis sonde linéaire entraînée sur encodeur gelé. Jeu de test : 3 065 images.

| Métrique     | Valeur |
|--------------|--------|
| Accuracy     | 0.8914 |
| AUC-ROC      | 0.9582 |
| Precision    | 0.9195 |
| Recall       | 0.8560 |
| F1-score     | 0.8866 |
| Specificity  | 0.9262 |

**Fidélité XAI** (AUC de la courbe de suppression — plus bas = plus fidèle) :
Grad-CAM 0.2249 · LIME 0.2253 · SHAP 0.2105.

Les figures et tableaux complets sont dans [`results/`](results/).

---

## Structure du dépôt

```
XAI-MedVision/
├── notebooks/            Pipeline complet (01 → 06)
├── src/                  Dataset & DataLoaders réutilisables
├── scripts/              Utilitaires (téléchargement des modèles)
├── data/
│   ├── processed/        Splits train/val/test (CSV) + config — versionnés
│   └── samples/          Petits échantillons 2D + 3D pour tester — versionnés
├── results/
│   ├── figures/          Figures générées par les notebooks
│   └── metrics/          Tableaux de métriques (CSV)
├── ui/                   Démo web (backend FastAPI + frontend React)
├── checkpoints/          Poids des modèles (non versionnés — voir plus bas)
└── requirements.txt      Dépendances du pipeline
```

> Les dossiers volumineux (`data/raw/`, `data/3d/`, `checkpoints/`) sont
> exclus de git. Voir les sections **Données** et **Modèles**.

---

## Modèles (checkpoints)

| Fichier | Taille | Contenu |
|---------|--------|---------|
| `checkpoints/ijepa/ijepa_best.pth` | ~353 Mo | Encodeur I-JEPA ViT-Small |
| `checkpoints/probe/probe_best.pth` | ~15 Ko  | Sonde linéaire |

L'encodeur dépasse la limite de fichier de GitHub (100 Mo). Les poids sont donc
hébergés sur **Hugging Face** :
[`zainabFarih/lung-ct-nodule-ijepa-vit-small`](https://huggingface.co/zainabFarih/lung-ct-nodule-ijepa-vit-small).

`python scripts/download_models.py` les place automatiquement dans
`checkpoints/ijepa/` et `checkpoints/probe/` — là où les notebooks **et** la
démo web les attendent. Ils peuvent aussi être regénérés via les notebooks 03 et 04.

---

## Échantillons de test

Pour tester sans télécharger les jeux complets, le dépôt inclut de petits
échantillons dans [`data/samples/`](data/samples/) :

- `2d/` — 6 images CT (3 saines, 3 avec nodule)
- `3d/` — 3 volumes CT réels NIfTI (`LIDC-IDRI-*.nii.gz`) pour la démo 3D

---

## Pipeline d'entraînement (notebooks)

Pour reproduire l'entraînement, installer les dépendances du pipeline :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |  Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
```

| Notebook | Rôle |
|----------|------|
| `01-eda-local.ipynb`            | Analyse exploratoire du jeu de données |
| `02_preprocessing_local.ipynb`  | Nettoyage, splits, normalisation, config |
| `03_ijepa_pretrain_local.ipynb` | Pré-entraînement auto-supervisé I-JEPA |
| `04_linear_probe_local.ipynb`   | Sonde linéaire sur encodeur gelé |
| `05_xai_comparison_local.ipynb` | Grad-CAM, LIME, SHAP + fidélité |
| `06_evaluation_local.ipynb`     | Évaluation finale et figures |

Chaque notebook détecte automatiquement la racine du projet — aucun chemin
absolu à modifier, ils s'adaptent à votre machine :

```python
ROOT = os.environ.get('PROJECT_ROOT') or (... détection automatique ...)
os.environ['PROJECT_ROOT'] = ROOT
```

---

## Données (entraînement)

Trop volumineuses pour git, à télécharger séparément.

### 2D — Classification (~20 000 images, ~1,4 Go)
Jeu de classification CT pulmonaire (`Healthy` / `Lung_Nodule`), sur Kaggle.
À placer dans :

```
data/raw/Lung_CT_Class_Dataset/Lung_CT_Class_Dataset/
├── Healthy/        healthy_XXXXX.png
└── Lung_Nodule/    nodule_XXXXX.png
```

Les splits `data/processed/{train,val,test}.csv` référencent ces fichiers en
chemins **relatifs** (`data/raw/...`) — ils fonctionnent dès que les images
sont à cet emplacement.

### 3D — Volumes CT (LIDC-IDRI, ~2,6 Go)
Collection **LIDC-IDRI** du *Cancer Imaging Archive (TCIA)*. À placer dans :

```
data/3d/
├── manifest-.../    séries DICOM
└── nifti/           volumes .nii.gz
```

> Vérifiez les conditions d'utilisation de chaque source avant téléchargement.

---

## Stack technique

I-JEPA · ViT-Small/16 (timm) · PyTorch · scikit-learn · Grad-CAM · LIME · SHAP
· FastAPI · React + Vite.
