# XAI-MedVision — Détection de nodules pulmonaires explicable

Détection de nodules pulmonaires sur images CT par **apprentissage auto-supervisé
(I-JEPA)** suivi d'une **sonde linéaire**, avec des explications visuelles
(**Grad-CAM, LIME, SHAP**) pour rendre chaque prédiction interprétable.

Le projet couvre toute la chaîne : EDA → prétraitement → pré-entraînement SSL →
sonde linéaire → comparaison XAI → évaluation, plus une **interface web de démo**
(FastAPI + React) qui accepte des images 2D et des volumes 3D.

---

## Table des matières

- [Démarrage rapide (démo web)](#démarrage-rapide-démo-web)
- [Démo en vidéo](#démo-en-vidéo)
- [Résultats](#résultats)
- [Comment les méthodes XAI expliquent les prédictions](#comment-les-méthodes-xai-expliquent-les-prédictions)
- [Structure du dépôt](#structure-du-dépôt)
- [Modèles (checkpoints)](#modèles-checkpoints)
- [Échantillons de test](#échantillons-de-test)
- [Pipeline d'entraînement (notebooks)](#pipeline-dentraînement-notebooks)
- [Données (entraînement)](#données-entraînement)
- [Stack technique](#stack-technique)

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

https://github.com/user-attachments/assets/a1815793-861d-4322-8e8c-011dd0555468



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

## Comment les méthodes XAI expliquent les prédictions

Le modèle prédit si une image CT contient un nodule pulmonaire, mais une prédiction seule ne suffit pas en contexte médical — il faut aussi savoir *pourquoi* le modèle a décidé ainsi. C'est le rôle des trois méthodes d'explicabilité intégrées : **Grad-CAM**, **LIME** et **SHAP**. Chacune répond à cette question d'une façon différente et complémentaire.

### Grad-CAM — Où le modèle regarde-t-il ?

Grad-CAM produit une carte de chaleur (*heatmap*) colorée superposée à l'image CT. Les zones rouges/orangées indiquent les régions qui ont le plus influencé la décision du modèle ; les zones bleues/froides ont peu ou pas contribué. Un clinicien peut ainsi vérifier en un coup d'œil si le modèle « regarde » au bon endroit — par exemple, si la zone chaude coïncide effectivement avec un nodule visible sur le scanner.

**Fonctionnement technique:**
- Calcul du gradient du score de prédiction par rapport aux cartes d'activation de la dernière couche de l'encodeur ViT.
- Ces gradients indiquent l'importance de chaque position spatiale ; ils sont moyennés par canal pour produire une carte d'importance 2D.
- La carte est ensuite redimensionnée à la taille de l'image d'origine par interpolation bilinéaire.

**Limites:**
- Résolution spatiale limitée par la taille des patches ViT (16 × 16 pixels).
- Indique *où* le modèle regarde, mais pas *comment* les valeurs de pixels individuels contribuent à la décision.

---

### LIME — Quelles zones de l'image sont déterminantes ?

LIME découpe l'image en petites régions homogènes appelées *superpixels*, puis masque certaines de ces régions et observe comment la probabilité de prédiction change. Les superpixels dont l'absence fait le plus baisser le score « nodule » sont colorés positivement (ils *soutiennent* la prédiction) ; ceux dont l'absence ne change rien, ou améliore le score, sont colorés négativement (ils *contredisent* ou sont neutres).

**Fonctionnement technique:**
- Génération d'un grand nombre de versions perturbées de l'image (superpixels aléatoirement masqués en gris).
- Récupération des prédictions du modèle sur chaque version perturbée.
- Entraînement d'un modèle linéaire local (régression Ridge) pondéré par la similarité de chaque perturbation avec l'image originale.
- Les coefficients de ce modèle linéaire constituent les scores d'importance par superpixel.

**Limites:**
- Les résultats peuvent légèrement varier d'une exécution à l'autre en raison du caractère aléatoire des perturbations.
- La segmentation en superpixels peut ne pas correspondre aux contours anatomiques réels du scanner.

---

### SHAP — Quelle est la contribution exacte de chaque pixel ?

SHAP attribue à chaque pixel (ou groupe de pixels) une valeur représentant sa contribution nette à l'écart entre la prédiction sur cette image et la prédiction moyenne du modèle sur l'ensemble des données. Un pixel avec une valeur SHAP positive pousse la prédiction vers « nodule » ; une valeur négative la pousse vers « sain ». Contrairement à Grad-CAM ou LIME, SHAP fournit une mesure rigoureusement justifiée par la théorie des jeux (valeurs de Shapley).

**Fonctionnement technique:**
- Utilisation de `DeepSHAP` (ou `GradientExplainer`), qui s'appuie sur les gradients du réseau et un ensemble d'images de référence (*background*).
- Pour chaque pixel, la valeur est calculée comme la différence moyenne d'activation entre l'image cible et les images de référence.
- Cette différence est pondérée par les gradients intégrés le long du chemin d'interpolation pour approximer les valeurs de Shapley.

**Limites:**
- Calcul plus coûteux que Grad-CAM ou LIME.
- Les explications pixel par pixel peuvent être difficiles à interpréter visuellement sans seuillage ni regroupement spatial.
- La qualité des explications dépend du choix des images de référence.

---

### Comparaison des méthodes et scores de fidélité

La **fidélité** mesure à quel point une explication reflète réellement le comportement du modèle. Elle est évaluée ici par la courbe de suppression (*deletion curve*) : on masque progressivement les pixels jugés les plus importants par chaque méthode et on observe la chute de confiance du modèle. **Un AUC plus bas indique une méthode plus fidèle** — les pixels qu'elle identifie sont effectivement ceux dont la suppression dégrade le plus la prédiction.

| Méthode   | AUC suppression | Interprétation |
|-----------|-----------------|----------------|
| Grad-CAM  | 0.2249          | Très fidèle — localisations précises et rapides |
| LIME      | 0.2253          | Très fidèle — résultats comparables à Grad-CAM |
| **SHAP**  | **0.2105**      | **Meilleure fidélité** — attributions les plus représentatives du modèle |

Les trois méthodes sont proches, ce qui suggère une cohérence globale des explications. SHAP est légèrement supérieur en fidélité, au prix d'un temps de calcul plus élevé. Dans la démo web, les trois sont disponibles simultanément pour permettre une comparaison visuelle directe.

> **Note pour les cliniciens.** Ces visualisations sont des outils d'aide à l'interprétation, pas des diagnostics. Une zone mise en évidence par le modèle doit toujours être confrontée à la lecture radiologique standard. Les méthodes XAI permettent de détecter des comportements inattendus du modèle (focalisation sur des artefacts, biais de fond), mais ne remplacent pas l'expertise médicale.



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
