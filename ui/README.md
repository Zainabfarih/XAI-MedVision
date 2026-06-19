# MedVision — Pulmonary AI Analysis

Interface web pour la détection de nodules pulmonaires avec explications visuelles (XAI).

- **backend** — API FastAPI (Python) : prédiction + Grad-CAM, LIME, SHAP
- **frontend** — Application React + Vite

## Entrées 2D et 3D (pseudo-3D)

Le modèle reste **2D** (coupes `224×224`). L'interface accepte deux types d'upload :

- **Image 2D** (`.png`, `.jpg`, `.tiff`) → prédiction + XAI sur la coupe, via `/api/predict` et `/api/explain/*`.
- **Volume 3D** (`.nii`, `.nii.gz`, `.dcm`/`.dicom`, `.npy`, `.npz`, TIFF multipage) → le backend découpe le volume en coupes 2D, applique le modèle 2D sur chaque coupe, puis agrège : verdict global, score par coupe, et coupe la plus suspecte. Endpoint : `/api/predict/volume`.

> En pseudo-3D, le modèle n'est pas réentraîné : c'est une boucle 2D sur les coupes. La coupe la plus suspecte alimente les vues Heatmap / Boundaries / Attribution.

## Prérequis

- Python 3.11 (PyTorch ne supporte pas encore 3.14)
- Node.js 18+

## Démarrage

Ouvrir **deux terminaux**.

### Terminal 1 — Backend

```cmd
cd ui\backend
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

- API : http://localhost:8000
- Statut : http://localhost:8000/api/status

> La création du venv et `pip install` ne se font qu'**une seule fois**.
> Ensuite, il suffit de `.venv\Scripts\activate` puis la commande `uvicorn`.

### Terminal 2 — Frontend

```cmd
cd ui\frontend
npm install
npm run dev
```

- Application : http://localhost:3000

> `npm install` ne se fait qu'**une seule fois**.

## Mode démo vs mode live

- **Mode live** : les checkpoints du modèle sont présents → vraies prédictions.
  `http://localhost:8000/api/status` affiche `"demo_mode": false`.
- **Mode démo** : pas de checkpoints → résultats simulés déterministes.
  L'interface reste entièrement fonctionnelle.

Pour le mode live, placer les deux fichiers entraînés dans `ui\backend\checkpoints\` :

```
ui\backend\checkpoints\ijepa_best.pth
ui\backend\checkpoints\probe_best.pth
```

Le backend les détecte automatiquement au démarrage.
