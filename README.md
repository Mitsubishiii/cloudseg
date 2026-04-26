# CloudSeg — Segmentation de nuages par deep learning sur imagerie satellitaire GOES-16

Projet de 2ème année à l'[ENSEA](https://www.ensea.fr), réalisé en collaboration avec l'[IGN](https://www.ign.fr).

**Auteurs** : Victor Fourel & Sacha Chérifi  
**Encadrant ENSEA** : Nicolas Simond  
**Encadrants IGN** : Marc Poupée, Arnaud Le Bris

---

## Présentation

Ce projet explore la **segmentation sémantique automatique de nuages** à partir des images du satellite géostationnaire **GOES-16** (NOAA). L'objectif est d'entraîner des modèles U-Net légers capables de reproduire — et de surpasser — une approche experte basée sur des règles physiques.

Le travail couvre l'ensemble de la chaîne :

1. Construction d'un dataset annoté par un algorithme expert (arbre de décision physique)
2. Analyse exploratoire (ACP, sélection de bandes spectrales)
3. Entraînement de deux variantes U-Net avec différentes combinaisons de bandes
4. Évaluation quantitative et comparaison des modèles
5. Application web interactive (Streamlit) pour tester les modèles en conditions réelles

---

## Structure du dépôt

```
cloudseg/
├── app.py                      # Application Streamlit (présentation + inférence live)
├── dataset_tool.py             # Assemblage et découpage du dataset multi-zones
├── CloudSegProject.ipynb       # Notebook : ACP, sélection de bandes, entraînement, évaluation
├── requirements.txt            # Dépendances Python
│
├── models/
│   ├── unet_best.pth           # Poids du Modèle 1 (3 bandes)
│   ├── unet_v2_best.pth        # Poids du Modèle 2 (4 bandes)
│   ├── vmin_3b.npy / vmax_3b.npy  # Bornes de normalisation (Modèle 1)
│   └── vmin_4b.npy / vmax_4b.npy  # Bornes de normalisation (Modèle 2)
│
└── images/
    ├── schema_pipeline_final_cut.jpg   # Schéma du pipeline de traitement
    ├── schema_data_final.jpg           # Schéma de construction du dataset
    ├── acp_variance.png / acp_classes.png / acp_3d.png  # Visualisations ACP
    ├── correlation_bandes.png          # Matrice de corrélation inter-bandes
    ├── score_F_bandes.png              # Scores F-ANOVA par bande
    ├── lda_classes.png                 # Analyse discriminante linéaire
    ├── eval_*M1 / eval_*M2             # Métriques d'évaluation (matrices de confusion,
    │                                   # courbes d'entraînement, IoU/F1, etc.)
    ├── timelapse_model_v1_*.gif        # Timelapse Modèle 1 (RGB / expert / U-Net)
    ├── timelapse_*M2.gif               # Timelapse Modèle 2
    ├── timelapse_v1_fusionne.mp4       # Vidéo comparative Modèle 1
    └── timelapse_v2_fusionne.mp4       # Vidéo comparative Modèle 2
```

---

## Données : satellite GOES-16

| Caractéristique | Valeur |
|---|---|
| Orbite | Géostationnaire (au-dessus des Amériques) |
| Capteur | ABI — Advanced Baseline Imager, 16 bandes |
| Résolution temporelle | Image plein-disque toutes les 10 minutes |
| Résolution spatiale utilisée | 4 km/pixel |
| Source des données | AWS S3 public — `noaa-goes16` |

**12 bandes utilisées** : C01–C06 (visible/proche-IR, 0.47–2.3 µm) + C07–C10 (IR moyen, 3.9–8.5 µm) + C13, C15 (IR thermique, 10.3–12.3 µm).

### Zones géographiques du dataset

| Zone | Région |
|---|---|
| Zone 1 | Golfe du Mexique |
| Zone 2 | Côte Est des États-Unis |
| Zone 3 | Caraïbes |

Le dataset est constitué de **patches 128×128 pixels**, découpés sur ces trois zones, puis divisés en 70 % entraînement / 15 % validation / 15 % test.

---

## Classes de nuages

9 classes de sortie (annotation par l'algorithme expert) :

| ID | Classe | Description |
|---|---|---|
| 0 | BG | Fond / pixels invalides |
| 1 | Surface | Ciel clair |
| 2 | Cumulus | Nuages bas convectifs |
| 3 | Stratus | Nuages bas stratiformes |
| 4 | Mid | Nuages moyens (250–273 K) |
| 5 | Cirrus | Nuages hauts de glace (transparents) |
| 6 | Cirrostratus | Nuages hauts de glace (épais) |
| 7 | Neige/Glace | Couverture neigeuse au sol |
| 8 | Cb | Cumulonimbus / orages |

L'approche experte repose sur trois indices physiques : **NDSI** (détection neige), **OTD** (convection profonde) et **SWD** (glace fine).

---

## Architecture des modèles

Les deux modèles partagent la même architecture **U-Net** légère :

- **Encodeur** : 4 niveaux, canaux [32, 64, 128, 256], MaxPool 2×2, Dropout progressif (0.1 → 0.5)
- **Goulot** : DoubleConv(256 → 512), Dropout 0.5
- **Décodeur** : 4 niveaux symétriques, ConvTranspose2d + skip connections
- **Sortie** : Conv 1×1 → 9 classes

Chaque bloc "DoubleConv" : Conv(3×3) → BN → ReLU → [Dropout] → Conv(3×3) → BN → ReLU

| | Modèle 1 | Modèle 2 |
|---|---|---|
| **Bandes d'entrée** | C13, C07, C04 | C13, C07, C04, C02 |
| **Type** | IR thermique + IR moyen + proche-IR | + canal rouge (visible) |
| **Taille modèle** | ~30 Mo | ~30 Mo |

---

## Résultats

| Métrique | Modèle 1 (3 bandes) | Modèle 2 (4 bandes) | Gain |
|---|---|---|---|
| **mIoU** | 0.588 | 0.756 | +28.5 % |
| **Macro-F1** | 0.720 | 0.858 | +19.1 % |
| IoU Surface | 0.86 | 0.93 | |
| IoU Stratus | — | — | Rappel : 0.60 → 0.98 |
| IoU Cumulus | 0.44 | 0.78 | |
| IoU Cumulonimbus | 0.90 | 0.94 | |

**Enseignement principal** : l'ajout de la bande C02 (rouge visible) est déterminant. Les nuages bas (Cumulus, Stratus) ont des signatures thermiques quasi-identiques — seule l'épaisseur optique, accessible via le visible, permet de les distinguer.

---

## Prérequis

- Python 3.9+
- GPU optionnel (CUDA) — l'inférence fonctionne sur CPU
- Accès Internet pour les données GOES-16 (AWS S3 public, sans authentification)

---

## Installation

```bash
git clone https://github.com/<votre-repo>/cloudseg.git
cd cloudseg
pip install -r requirements.txt
```

Dépendances principales :

```
torch / torchvision     # Deep learning
streamlit               # Application web
satpy / pyresample      # Traitement données satellitaires
s3fs                    # Accès AWS S3
netCDF4 / h5netcdf      # Format NetCDF (données GOES)
numpy / pandas          # Manipulation de données
matplotlib / plotly     # Visualisation
imageio / imageio-ffmpeg # Export vidéo
```

---

## Reproduire les résultats

### 1. Préparer le dataset

Le dataset doit être construit depuis les données brutes GOES-16. Le script `dataset_tool.py` assemble et découpe les zones géographiques :

```bash
python dataset_tool.py
```

Il attend des fichiers `X_<zone>.npy` / `y_<zone>.npy` pour chacune des 3 zones, et produit :

```
dataset_multizone_final/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
```

### 2. Entraînement et analyse

Ouvrir et exécuter le notebook `CloudSegProject.ipynb` :

```bash
jupyter notebook CloudSegProject.ipynb
```

Le notebook couvre dans l'ordre :
1. Chargement et visualisation des données depuis AWS S3
2. Analyse en composantes principales (ACP) sur les 12 bandes
3. Sélection des bandes par corrélation et scores F-ANOVA
4. Entraînement du Modèle 1 (3 bandes) et du Modèle 2 (4 bandes)
5. Évaluation sur le jeu de test (matrices de confusion, IoU, F1, courbes d'apprentissage)
6. Génération des timelapses de prédiction

Les poids entraînés sont sauvegardés dans `models/`.

---

## Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur sur `http://localhost:8501`.

Elle propose 6 onglets :

| Onglet | Contenu |
|---|---|
| **Introduction** | Contexte, satellite GOES-16, schéma du pipeline |
| **Approche experte & Dataset** | Règles physiques, indices spectraux, exemples |
| **ACP & Sélection** | Variance expliquée, séparabilité, corrélations, F-scores |
| **Modèle 1 (3B)** | Courbes d'entraînement, métriques, points forts/faibles |
| **Modèle 2 (4B)** | Idem + analyse du gain apporté par C02 |
| **Test live** | Inférence sur données GOES-16 en temps réel via AWS S3 |

L'onglet **Test live** permet de :
- Télécharger une scène GOES-16 directement depuis S3
- Comparer côte à côte l'image RGB, la classification experte et les deux U-Net
- Explorer des patches aléatoires
- Générer une vidéo comparative sur 5 heures du cycle diurne

L'interface est disponible en **français et en anglais** (bouton en haut à droite).

---

## Licence

Projet académique — ENSEA / IGN, 2025–2026. Usage libre pour la recherche et l'enseignement.
