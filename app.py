import base64
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import io
import tempfile
import imageio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.io as pio

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Dashboard IA | GOES-16", layout="centered")

# ==========================================
# 1. ARCHITECTURE U-NET
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=9, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            drop = 0.1 if f <= 64 else 0.4
            self.encoders.append(DoubleConv(ch, f, dropout=drop))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.5)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            drop = 0.1 if f <= 64 else 0.3
            self.decoders.append(DoubleConv(f * 2, f, dropout=drop))
            ch = f
        self.head = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.head(x)

# ==========================================
# 2. CONSTANTES ET PALETTE EXPERTE
# ==========================================
INDICES_3B = [10, 6, 3]      # C13, C07, C04
INDICES_4B = [10, 6, 3, 1]   # C13, C07, C04, C02
PATCH_SIZE = 128

COULEURS_CLASSES = ['#000000', '#8B4513', '#87CEEB', '#D3D3D3', '#A9A9A9', '#00FFFF', '#FFFFFF', '#0000FF', '#FF00FF']
CLASS_NAMES = {0: 'BG', 1: 'Surface', 2: 'Cumulus', 3: 'Stratus', 4: 'Mid', 5: 'Cirrus', 6: 'Cirrostratus', 7: 'Snow', 8: 'Cb'}
CMAP_EXPERT = ListedColormap(COULEURS_CLASSES)
NORM_EXPERT = BoundaryNorm(np.arange(-0.5, 9.5, 1), CMAP_EXPERT.N)
LEGEND_PATCHES = [mpatches.Patch(color=COULEURS_CLASSES[c], label=CLASS_NAMES[c]) for c in range(1, 9)]

# ==========================================
# 3. FONCTIONS UTILITAIRES (MOTEUR)
# ==========================================
def build_rgb(img_12b):
    R = np.clip(np.nan_to_num(img_12b[..., 1]), 0, 1)
    NIR = np.clip(np.nan_to_num(img_12b[..., 2]), 0, 1)
    B = np.clip(np.nan_to_num(img_12b[..., 0]), 0, 1)
    G = np.clip(0.45*R + 0.10*NIR + 0.45*B, 0, 1)
    return np.clip(np.sqrt(np.dstack([R, G, B])), 0, 1)

@torch.no_grad()
def infer_full_zone(model, img_12b, indices_sel, vmin, vmax, device):
    """
    Pipeline exact : 
    1. Traitement physique (Visible 0-1 / IR Kelvin)
    2. Sélection des bandes (3 ou 4)
    3. Normalisation Min-Max globale
    4. Tiling (128x128)
    """
    H, W, _ = img_12b.shape
    X_phys = np.zeros_like(img_12b, dtype=np.float32)
    
    # --- ÉTAPE 1 : Traitement physique identique à build_X_y ---
    for i in range(12):
        data = np.nan_to_num(img_12b[..., i])
        if i <= 5: # Bandes visibles C01 à C06
            if np.max(data) > 2.0: data /= 100.0
            data = np.clip(data, 0, 1)
        X_phys[..., i] = data
        
    # --- ÉTAPE 2 & 3 : Sélection et Normalisation Min-Max ---
    X_sel = X_phys[..., indices_sel]
    X_norm = np.clip((X_sel - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    # --- ÉTAPE 4 : Inférence par tuiles ---
    y_pred = np.zeros((H, W), dtype=np.uint8)
    for i in range(0, H, PATCH_SIZE):
        for j in range(0, W, PATCH_SIZE):
            i_end, j_end = min(i + PATCH_SIZE, H), min(j + PATCH_SIZE, W)
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE, len(indices_sel)), dtype=np.float32)
            actual_p = X_norm[i:i_end, j:j_end, :]
            patch[:actual_p.shape[0], :actual_p.shape[1], :] = actual_p
            
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(t).argmax(dim=1).squeeze().cpu().numpy()
            y_pred[i:i_end, j:j_end] = pred[:i_end-i, :j_end-j]
            
    return y_pred

def fig_to_array(fig):
    # Force le rendu pour éviter la première frame noire
    fig.canvas.draw()
    buf = io.BytesIO()
    # Utilisation d'une résolution fixe pour éviter les décalages de pixels
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = imageio.imread(buf)
    return img[..., :3]

def svg_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def autoplay_video(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f"""
        <video width="100%" autoplay loop muted playsinline controls>
            <source src="data:video/mp4;base64,{data}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# ==========================================
# EN-TÊTE | Logos SVG + HTML
# ==========================================

try:
    logo1_b64 = svg_to_base64("images/logo_ensea.svg")
    logo2_b64 = svg_to_base64("images/logo_geodata.svg")
    logos_html = f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 60px; margin-bottom: 10px;">
        <img src="data:image/svg+xml;base64,{logo1_b64}" style="height: 80px;">
        <img src="data:image/svg+xml;base64,{logo2_b64}" style="height: 80px;">
    </div>
    """
except Exception as e:
    logos_html = f"<p style='text-align:center; color:red;'>Logos introuvables : {e}</p>"

html_header = f"""
{logos_html}
<div style="text-align: center; margin-top: 15px;">
    <h1 style="margin-bottom: 0px;">Application de l'IA à l'Imagerie Satellitaire</h1>
    <h3 style="margin-top: 5px; color: #555555;">Projet de 2ème Année | Approche Experte et Réduction de Dimension</h3>
    <br>
    <p style="font-size: 18px; margin-bottom: 0px;">
        <strong>Victor Fourel</strong> &amp; <strong>Sacha Chérifi</strong>
    </p>
    <p style="font-size: 14px; margin-top: 0px;">
        <i>École Nationale Supérieure de l'Électronique et de ses Applications (ENSEA)</i>
    </p>
    <br>
    <p style="font-size: 16px; margin-bottom: 0px;">
        <strong>Encadrant ENSEA :</strong> M. Nicolas Simond<br>
        <strong>Encadrants IGN :</strong> M. Marc Poupée et M. Arnaud Le Bris
    </p>
    <br>
    <p style="font-size: 14px; color: #888888;">Année 2026</p>
</div>
<hr>
"""

st.markdown(html_header, unsafe_allow_html=True)

# ==========================================
# FONCTION DE MISE EN PAGE
# ==========================================
def afficher_section_centree(titre, image_path, texte_analyse, type_message="info"):
    st.markdown(f"<h3 style='text-align: center;'>{titre}</h3>", unsafe_allow_html=True)
    
    try:
        st.image(f"images/{image_path}", use_container_width=True)
    except Exception as e:
        st.error(f"Image introuvable : images/{image_path}")
        
    if type_message == "info":
        st.info(texte_analyse)
    elif type_message == "warning":
        st.warning(texte_analyse)
    elif type_message == "success":
        st.success(texte_analyse)
        
    st.markdown("---")

# ==========================================
# CRÉATION DES ONGLETS
# ==========================================
tab0, tab2_expert, tab3_acp, tab4, tab5, tab6 = st.tabs([
    "Introduction",
    "Approche Experte & Dataset",
    "ACP & Sélection",
    "Modèle 1 3B",
    "Modèle 2 4B",
    "Test en direct (Inférence)",
])

# ------------------------------------------
# ONGLET 0 : INTRODUCTION & PROBLÉMATIQUE
# ------------------------------------------
with tab0:
    st.header("Introduction & Problématique")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#1e3a5f; padding: 20px; border-radius: 10px; border-left: 5px solid #4a9eda;">
        <h4 style="color:#4a9eda; margin-top:0;">Problématique</h4>
        <p style="font-size:16px; color:white;">
            Comment reproduire un arbre de décision météorologique complexe à partir d'images 
            satellites géostationnaires en utilisant des réseaux de neurones allégés ?
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_sat, col_disk = st.columns(2)
    with col_sat:
        st.markdown("<h4 style='text-align:center;'>Satellite GOES-16 (NOAA)</h4>", unsafe_allow_html=True)
        try:
            st.image("images/satellite_goes16.png", use_container_width=True)
        except:
            st.info("Image satellite_goes16.png non trouvée")

    with col_disk:
        st.markdown("<h4 style='text-align:center;'>Vue Disque Entier (RGB Synthétique)</h4>", unsafe_allow_html=True)
        try:
            st.image("images/full_disk.jpg", use_container_width=True)
        except:
            st.info("Image full_disk.jpg non trouvée")

    st.markdown("---")

    st.markdown("""
    <h4>🛰️ Le Satellite GOES-16</h4>
    <ul style="font-size:15px; line-height:2;">
        <li>Orbite <strong>géostationnaire</strong> | observation continue de l'Amérique</li>
        <li>Capteur <strong>ABI (Advanced Baseline Imager)</strong> : 16 bandes spectrales du visible à l'infrarouge thermique</li>
        <li>Images toutes les <strong>10 minutes</strong> sur le disque entier</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h4 style='text-align:center;'>Pipeline de Traitement</h4>", unsafe_allow_html=True)
    try:
        st.image("images/schema_pipeline_final_cut.jpg", use_container_width=True)
    except:
        st.info("Image schema_pipeline_final_cut.jpg non trouvée")

    st.markdown("---")

    st.markdown("<h4 style='text-align:center;'>Reprojection Géométrique</h4>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="font-size:15px; line-height:2;">
        <li>Le satellite observe la Terre comme un <strong>disque 3D déformé</strong> sur les bords</li>
        <li>Les pixels bruts sont projetés sur une <strong>grille géographique 2D régulière</strong> via rééchantillonnage</li>
    </ul>
    """, unsafe_allow_html=True)
    try:
        st.image("images/schema_data_final.jpg", use_container_width=True)
        st.caption("Transformation spatiale et rééchantillonnage des pixels")
    except:
        st.info("Image schema_data_final.jpg non trouvée")


# ------------------------------------------
# ONGLET 1 : APPROCHE EXPERTE & DATASET
# ------------------------------------------
with tab2_expert:
    st.header("Approche Experte & Génération du Dataset")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Les 8 bandes ---
    st.markdown("<h3 style='text-align:center;'>Les 8 Bandes GOES-16 Utilisées</h3>", unsafe_allow_html=True)

    import pandas as pd
    df_bandes = pd.DataFrame({
        "Bande":         ["C02", "C03", "C04", "C05", "C06", "C08", "C13", "C15"],
        "Caractéristique": [
            "Rouge | Épaisseur optique",
            "Végétation | Contraste Terre/Mer",
            "Proche-IR | Détection Cirrus",
            "Proche-IR | Absorption Neige",
            "Microphysique | Taille des particules",
            "IR | Vapeur d'eau haute",
            "IR | Température du sommet nuageux",
            "IR Sale | Sensibilité glace/humidité"
        ]
    })
    st.table(df_bandes)

    st.markdown("---")

    # --- Feature Engineering ---
    st.markdown("<h3 style='text-align:center;'>Feature Engineering : Les Indices Physiques</h3>", unsafe_allow_html=True)

    col_ndsi, col_otd, col_swd = st.columns(3)
    with col_ndsi:
        st.markdown("""
        <div style="background:#1a3a1a; padding:15px; border-radius:8px; border-left:4px solid #4CAF50;">
            <h5 style="color:#4CAF50; margin-top:0;">NDSI</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">
                (C02 − C05) / (C02 + C05)
            </p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#4CAF50;">Cible : Neige</strong><br>
                Forte réflexion C02, forte absorption C05.<br>
                NDSI > 0.4 → neige détectée
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_otd:
        st.markdown("""
        <div style="background:#1a1a3a; padding:15px; border-radius:8px; border-left:4px solid #4a9eda;">
            <h5 style="color:#4a9eda; margin-top:0;">OTD</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">
                C08 − C13
            </p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#4a9eda;">Cible : Convection</strong><br>
                Dépassement stratosphérique.<br>
                OTD > −5.0 → Cumulonimbus
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_swd:
        st.markdown("""
        <div style="background:#3a1a1a; padding:15px; border-radius:8px; border-left:4px solid #e74c3c;">
            <h5 style="color:#e74c3c; margin-top:0;">SWD</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">
                C13 − C15
            </p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#e74c3c;">Cible : Glace fine</strong><br>
                Discrimine Cirrus vs nuage opaque.<br>
                SWD > 1.5 K → Cirrus/Cs
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Arbre de décision ---
    st.markdown("<h3 style='text-align:center;'>L'Arbre Décisionnel Expert | Règles Logiques</h3>", unsafe_allow_html=True)

    df_arbre = pd.DataFrame({
        "Classe": [
            "Cumulonimbus",
            "Snow/Ice",
            "Cirrostratus",
            "Cirrus",
            "Mid-Level",
            "Stratus",
            "Cumulus",
            "Surface"
        ],
        "Conditions Physiques Principales": [
            "T₁₃ < 235 K  ET  OTD > −5.0",
            "NDSI > 0.4  ET  T₁₃ < 273 K",
            "Glace  ET  T₁₃ < 260 K  ET  C02 > 0.20",
            "Glace  ET  voile semi-transparent (C02 < 0.35)",
            "Nuage eau  ET  250 K ≤ T₁₃ < 273 K",
            "Nuage eau bas (T₁₃ ≥ 273 K)  ET  C02 > 0.45",
            "Nuage eau bas (T₁₃ ≥ 273 K)  ET  C02 ≤ 0.45",
            "Aucun seuil nuageux ou neigeux atteint"
        ]
    })
    st.table(df_arbre)

    st.markdown("---")

    # --- Construction du dataset ---
    st.markdown("<h3 style='text-align:center;'>Construction du Dataset</h3>", unsafe_allow_html=True)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("""
        <div style="background:#1e2a1e; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#4CAF50;">128 × 128</h2>
            <p style="color:#aaa;">Taille des patches<br>(pixels)</p>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown("""
        <div style="background:#1e1e2a; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#4a9eda;">3 Zones</h2>
            <p style="color:#aaa;">Golfe du Mexique<br>Côte Est USA · Caraïbes</p>
        </div>
        """, unsafe_allow_html=True)
    with col_d3:
        st.markdown("""
        <div style="background:#2a1e1e; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#e74c3c;">70/15/15</h2>
            <p style="color:#aaa;">Split<br>Train / Val / Test</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.warning("""
    **⚠️ Points de vigilance majeurs :**
    - **Gestion du vide :** Exclusion des bords spatiaux (artefacts de projection à 0 ou NaN)
    - **Déséquilibre des classes :** La classe Surface est omniprésente, contrairement aux Cumulonimbus
    - **Normalisation :** Bornes physiques absolues pour ramener IR (Kelvin) et Visible (Réflectances) sur [0, 1] sans Data Shift
    """)

    st.markdown("---")
    st.markdown("<h4 style='text-align:center;'>Exemples de Classification Experte</h4>", unsafe_allow_html=True)
    try:
        st.image("images/cas_extremes_expert2_comp.png", use_container_width=True)
    except:
        st.info("Image cas_extremes_expert2_comp.png non trouvée")


# ------------------------------------------
# ONGLET 2 : ACP & SÉLECTION DES BANDES
# ------------------------------------------
with tab3_acp:
    st.header("Analyses Statistiques & Sélection des Bandes")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- ACP ---
    st.markdown("<h3 style='text-align:center;'>Analyse en Composantes Principales (ACP)</h3>", unsafe_allow_html=True)
    st.info("""
    **Objectif :** Identifier la redondance dans les 12 bandes et réduire l'espace des caractéristiques d'entrée du modèle.
    """)

    df_acp = pd.DataFrame({
        "Composante": ["PC1", "PC2", "PC3"],
        "Information capturée": [
            "Dynamique thermique globale | nuages hauts vs surface",
            "Albédo et réflectances | épaisseur optique des nuages",
            "Phénomènes marginaux de transition | glace/eau"
        ]
    })
    st.table(df_acp)

    afficher_section_centree(
        "Scree Plot & Loadings ACP",
        "acp_variance.png",
        "PC1 et PC2 capturent l'essentiel de la variance. Les bandes thermiques dominent PC1, les bandes visibles dominent PC2."
    )

    afficher_section_centree(
        "Visualisation 2D de l'Espace Latent",
        "acp_classes.png",
        "Séparabilité partielle des classes dans le plan PC1/PC2. Les classes extrêmes (Surface, Cumulonimbus) sont bien isolées."
    )

    # afficher_section_centree(
    #     "Visualisation 3D de l'Espace Latent",
    #     "acp_3d.png",
    #     "PC3 apporte une séparation supplémentaire pour les nuages de glace (Cirrus, Cirrostratus) qui se superposent sur les 2 premiers axes."
    # )

    import plotly.io as pio
    import plotly.graph_objects as go

    st.markdown(f"<h3 style='text-align: center;'>{"Séparabilité des classes (ACP 3D)"}</h3>", unsafe_allow_html=True)

    # Solution : Lire le fichier en tant que texte et utiliser from_json avec skip_invalid
    try:
        with open("./images/acp_3d_figure.json", "r") as f:
            json_clean = f.read()
        
        # On force Plotly à ignorer les erreurs de propriétés comme 'heatmapgl'
        fig = pio.from_json(json_clean, skip_invalid=True)
        
        # Affichage interactif pleine largeur
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement de l'ACP 3D : {e}")

    st.info("PC3 apporte une séparation supplémentaire pour les nuages de glace (Cirrus, Cirrostratus) qui se superposent sur les 2 premiers axes.")


    # --- Matrice de corrélation ---
    st.markdown("<h3 style='text-align:center;'>Matrice de Corrélation de Pearson</h3>", unsafe_allow_html=True)
    afficher_section_centree(
        "Corrélations Inter-Bandes",
        "correlation_bandes.png",
        "Démarcation franche entre deux familles : C01→C06 (Visible/Proche-IR, très corrélées) et C07→C15 (Infrarouge Thermique). Éviter les paires redondantes en entrée du modèle."
    )

    st.markdown("---")

    # --- Score F-ANOVA ---
    st.markdown("<h3 style='text-align:center;'>Pouvoir Discriminant : Score F-ANOVA</h3>", unsafe_allow_html=True)
    afficher_section_centree(
        "Score F-ANOVA par Bande",
        "score_F_bandes.png",
        "Le Score F mesure le rapport variance inter-classes / variance intra-classes. Plus F est élevé, mieux la bande discrimine les 8 classes nuageuses."
    )


    # --- Synthèse sélection ---
    st.markdown("<h3 style='text-align:center;'>Synthèse : Sélection des Bandes Optimales</h3>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown("""
        <div style="background:#1e2a3a; padding:20px; border-radius:8px; border-top:4px solid #4a9eda; text-align:center;">
            <h3 style="color:#4a9eda;">C13</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">Infrarouge Thermique</strong><br>
                Majorité de la variance (PC1)<br>
                → Altitude des sommets nuageux
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b2:
        st.markdown("""
        <div style="background:#2a1e3a; padding:20px; border-radius:8px; border-top:4px solid #9b59b6; text-align:center;">
            <h3 style="color:#9b59b6;">C07</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">Ondes Courtes IR</strong><br>
                Proxy statistique brouillards/stratus<br>
                → Absente de l'arbre expert !
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b3:
        st.markdown("""
        <div style="background:#1a3a2a; padding:20px; border-radius:8px; border-top:4px solid #4CAF50; text-align:center;">
            <h3 style="color:#4CAF50;">C04</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">Proche Infrarouge</strong><br>
                Information orthogonale (PC2)<br>
                → Distinction des Cirrus fins
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.success("""
    **Le raccourci mathématique de l'IA**  
    L'IA ne copie pas la logique humaine. Ces 3 bandes suffisent à retrouver l'information latente 
    exploitée par l'arbre expert à 8 bandes | avec en plus C02 pour le Modèle 2 (+28.5% mIoU).
    """)

# ------------------------------------------
# ONGLET MODÈLE 1
# ------------------------------------------
with tab4:
    st.header("Analyse du Modèle 1 (3 Bandes : C13, C07, C04 | 30 Epochs)")
    st.markdown("<br>", unsafe_allow_html=True)

    afficher_section_centree(
        "Courbes d'entraînement",
        "eval_4_training_curves.png",
        "La Train Loss descend régulièrement de 1.35 à 0.60, signe d'un apprentissage stable. "
        "La Val Loss converge vers 0.52 sans diverger | pas d'overfitting. "
        "En revanche le mIoU Val oscille fortement (0.38 → 0.55) sans jamais se stabiliser, "
        "révélant que le modèle n'a pas encore trouvé de frontières de décision stables entre classes proches."
    )

    afficher_section_centree(
        "Matrice de Confusion",
        "eval_1_confusion_matrix.png",
        "Deux points forts nets : Surface (0.86) et Cumulonimbus (0.90), classes aux signatures spectrales extrêmes. "
        "Le point faible majeur est la confusion Stratus/Cumulus : 32% des Stratus sont classés Cumulus "
        "et 18% des Cumulus sont classés Stratus | ces deux classes bas ont des signatures thermiques "
        "quasi-identiques sans la bande visible C02. "
        "Snow/Ice (0.75) confond 15% de ses pixels avec Cirrostratus, ce qui est physiquement cohérent "
        "(deux surfaces froides et réfléchissantes)."
    )

    afficher_section_centree(
        "Distribution de classe",
        "eval_5_distribution.png",
        "Biais de sous-estimation de la Surface (63% réel → 56% prédit) compensé par une surestimation "
        "du Cumulus (9% → 13%) et du Stratus (1.5% → 3%). "
        "Le modèle 'hésite' et sur-segmente les nuages bas au détriment de la surface dégagée. "
        "Cirrus et Cumulonimbus sont bien calibrés (écart < 1%).",
        type_message="warning"
    )

    afficher_section_centree(
        "Radar de performance",
        "eval_6_radar.png",
        "Profil en 'comète' très caractéristique : deux pointes vers Surface et Cumulonimbus "
        "(IoU > 0.80), puis un effondrement brutal sur Cumulus (IoU ~0.36) et Stratus (IoU ~0.25). "
        "Le fossé entre Recall (orange) et IoU (bleu) est particulièrement large sur Stratus et Cumulus, "
        "confirmant que le modèle détecte ces classes mais avec beaucoup de faux positifs.",
        type_message="warning"
    )

    afficher_section_centree(
        "Métriques par classe",
        "eval_2_iou_f1_precision_recall.png",
        "mIoU global de 0.588 et Macro-F1 de 0.720. "
        "Écart de performance extrême : Surface (IoU=0.85, F1=0.92) vs Stratus (IoU=0.25, F1=0.40). "
        "Stratus et Cumulus ont une Precision faible (<0.45) mais un Recall correct (>0.60), "
        "signe classique d'un modèle qui sur-prédit ces classes pour 'ratisser large'. "
        "Cumulonimbus est la surprise positive avec IoU=0.82 et F1=0.90 malgré sa rareté.",
        type_message="warning"
    )

    afficher_section_centree(
        "Dice Scores",
        "eval_3_dice.png",
        "La heatmap confirme le diagnostic : Surface (0.916) et Cumulonimbus (0.899) en vert foncé, "
        "Stratus (0.405) en jaune-orange et Cumulus (0.530) en jaune pâle. "
        "Ces deux classes représentent le plafond de performance du Modèle 1 | "
        "sans bande visible, l'épaisseur optique des nuages bas est invisible pour le réseau.",
        type_message="warning"
    )

    afficher_section_centree(
        "Prédictions vs Réel (Modèle 1)",
        "PredictionvsterrainM1.png",
        "Visuellement, la segmentation spatiale est cohérente sur les grandes structures "
        "(zones de surface, systèmes convectifs). Les erreurs se concentrent aux frontières "
        "Cumulus/Stratus où le modèle produit des transitions floues au lieu de limites nettes."
    )

    st.markdown("<h3 style='text-align: center;'>Comparaison Dynamique | Timelapse Modèle 1</h3>",
                unsafe_allow_html=True)
    autoplay_video("images/timelapse_v1_fusionne.mp4")
    st.info("Sur le timelapse, on observe la montée en puissance des Cumulonimbus l'après-midi "
            "(13h→19h UTC) correctement capturée. En revanche les zones de stratus marins "
            "en bordure de zone sont régulièrement confondues avec du Cumulus.")
    st.warning("**Résumé Modèle 1 :** Base solide sur les phénomènes à fort contraste thermique. "
               "Le plafond à mIoU=0.588 est structurel : sans bande visible, "
               "Stratus et Cumulus sont spectralement indiscernables pour le réseau.")
    
# ------------------------------------------
# ONGLET MODÈLE 2
# ------------------------------------------
with tab5:
    st.header("Analyse du Modèle 2 (4 Bandes : C13, C07, C04, C02 | 30 Epochs)")
    st.markdown("<br>", unsafe_allow_html=True)

    afficher_section_centree(
        "Courbes d'entraînement",
        "eval_4_training_curvesM2.png",
        "Saut qualitatif immédiat dès l'epoch 1 : la Val Loss part de 0.80 et converge vers 0.27, "
        "soit une réduction de 48% vs le Modèle 1. "
        "Le mIoU Val monte de façon bien plus stable (0.55 → 0.72+) avec des oscillations réduites | "
        "preuve que C02 (bande visible rouge) fournit un signal discriminant fort et stable "
        "que le réseau exploite dès les premières epochs.",
        type_message="success"
    )

    afficher_section_centree(
        "Matrice de Confusion",
        "eval_1_confusion_matrixM2.png",
        "Transformation radicale. Stratus passe de 0.60 à 0.98 de rappel | quasi-disparition des erreurs. "
        "Cumulus atteint 0.93, Surface 0.94, Cumulonimbus 0.93. "
        "La seule faiblesse résiduelle : Snow/Ice (0.78) confond encore 11% de ses pixels "
        "avec Cirrostratus | confusion physiquement explicable (deux surfaces froides "
        "à haute réflectance en C02).",
        type_message="success"
    )

    afficher_section_centree(
        "Distribution de classe",
        "eval_5_distributionM2.png",
        "Alignement quasi parfait entre distributions réelle et prédite pour toutes les classes. "
        "Le biais de sous-estimation de la Surface est réduit à seulement 3% (63→60%), "
        "contre 7% pour le Modèle 1. "
        "Le sur-étiquetage Cumulus a quasiment disparu (9%→10% vs 9%→13% en M1). "
        "C02 a fourni l'information d'épaisseur optique qui manquait pour calibrer correctement "
        "les classes de nuages bas.",
        type_message="success"
    )

    afficher_section_centree(
        "Radar de performance",
        "eval_6_radarM2.png",
        "Transformation du profil 'comète' en profil quasi-circulaire. "
        "Toutes les classes sont désormais entre IoU=0.62 (Mid-Level) et IoU=0.94 (Surface). "
        "Le fossé Recall/IoU s'est refermé sur toutes les classes | le modèle ne sur-prédit plus. "
        "Seul Snow/Ice reste légèrement en retrait (IoU~0.62) à cause de la confusion avec Cirrostratus.",
        type_message="success"
    )

    afficher_section_centree(
        "Métriques par classe",
        "eval_2_iou_f1_precision_recallM2.png",
        "mIoU de 0.756 (+28.5%) et Macro-F1 de 0.858 (+19.1%). "
        "Stratus : IoU 0.25→0.75 (+200%), F1 0.40→0.88 (+120%). "
        "Cumulus : IoU 0.36→0.77 (+114%). "
        "La classe la plus difficile reste Mid-Level (IoU=0.65) | altitude intermédiaire "
        "difficile à distinguer spectralement des Cirrus et Stratus épais.",
        type_message="success"
    )

    afficher_section_centree(
        "Dice Scores",
        "eval_3_diceM2.png",
        "Toutes les classes dépassent le seuil 0.76 : Surface (0.968), Cumulonimbus (0.905), "
        "Stratus (0.881), Cumulus (0.874), Cirrus (0.845), Cirrostratus (0.838), "
        "Mid-Level (0.785), Snow/Ice (0.766). "
        "La bande C02 a résolu structurellement la faiblesse Stratus/Cumulus "
        "qui plafonnait le Modèle 1.",
        type_message="success"
    )

    afficher_section_centree(
        "Prédictions vs Réel (Modèle 2)",
        "unet_predictions_rgbM2.png",
        "Fidélité remarquable à l'arbre expert. Les frontières entre classes sont nettes "
        "et les structures nuageuses fines (cirrus filamenteux, bords de Cumulonimbus) "
        "sont correctement délimitées. La surface dégagée est parfaitement isolée "
        "même en présence de nuages translucides.",
        type_message="success"
    )

    st.markdown("<h3 style='text-align: center;'>Comparaison Dynamique | Timelapse Modèle 2</h3>",
                unsafe_allow_html=True)
    autoplay_video("images/timelapse_v2_fusionne.mp4")
    st.success("Stabilité temporelle nettement supérieure. Les transitions entre frames "
               "sont cohérentes et sans artefacts de classification. "
               "L'évolution du cycle convectif diurne (formation→maturité→dissipation des Cb) "
               "est fidèlement reproduite.")
    st.success("**Résumé Modèle 2 :** L'ajout de C02 (canal rouge haute résolution) "
               "apporte l'information d'épaisseur optique manquante. "
               "Le réseau atteint mIoU=0.756 en égalant la logique de l'arbre expert "
               "sur 7 des 8 classes.")

# ------------------------------------------
# ONGLET 3 : TEST EN DIRECT (INFERENCE)
# ------------------------------------------
# ==========================================
# 2. FONCTIONS DE TRAITEMENT
# ==========================================

import os, gc, s3fs
import numpy as np
from datetime import datetime
from satpy import Scene
from pyresample import create_area_def
from pyproj import CRS, Transformer

RESOLUTION  = 4000               # 4km : Indispensable pour la légèreté
GOES_DIR    = './goes_images'
BANDS       = ['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C13','C15']
CLASS_MAP   = {'Surface':1,'Cumulus':2,'Stratus':3,'Mid-Level':4,'Cirrus':5,'Cirrostratus':6,'Snow/Ice':7,'Cumulonimbus':8}

def get_area_definition(zone_coords, resolution=4000):
    proj = {'proj': 'eqc', 'datum': 'WGS84', 'lat_ts': 0, 'lon_0': 0}
    t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_dict(proj), always_xy=True)
    xmin, ymin = t.transform(zone_coords[0], zone_coords[1])
    xmax, ymax = t.transform(zone_coords[2], zone_coords[3])
    return create_area_def('zone_custom', projection=proj, area_extent=[xmin, ymin, xmax, ymax],
                           resolution=resolution, units='meters')

def get_expert_masks(scn):
    def to_phys(band):
        data = np.nan_to_num(scn[band].values)
        if 'C01' <= band <= 'C06':
            if np.max(data) > 2.0: data /= 100.0
            return np.clip(data, 0, 1)
        return data
    r_vis=to_phys('C02'); r_cir=to_phys('C04'); r_snw=to_phys('C05')
    t_wv=to_phys('C08');  t_win=to_phys('C13'); t_dirty=to_phys('C15')
    ndsi=(r_vis-r_snw)/(r_vis+r_snw+1e-6); otd=t_wv-t_win; swd=t_win-t_dirty
    m_cb=(t_win<235)&(otd>-5.0); m_sn=(ndsi>0.4)&(t_win<273)&(~m_cb)
    is_ice=(r_cir>0.02)|((swd>1.5)&(t_win<273))
    m_cs=is_ice&(t_win<260)&(r_vis>0.20)&(~m_cb)&(~m_sn)
    m_ci=is_ice&(~m_cs)&(r_vis<0.35)&(~m_cb)&(~m_sn)
    is_water=(r_vis>0.15)&(~m_cb)&(~m_sn)&(~m_cs)&(~m_ci)
    m_mid=is_water&(t_win<273)&(t_win>=250); is_low=is_water&(~m_mid)
    m_st=is_low&(r_vis>0.45); m_cu=is_low&(r_vis<=0.45)
    m_sfc=~(m_cb|m_sn|m_cs|m_ci|m_mid|m_st|m_cu)
    return [('1. Surface',m_sfc),('2. Snow/Ice',m_sn),('3. Cumulonimbus',m_cb),
            ('4. Cirrostratus',m_cs),('5. Cirrus',m_ci),('6. Mid-Level',m_mid),
            ('7. Stratus',m_st),('8. Cumulus',m_cu)]

def load_sat_data(date, zone_name):
    fs = s3fs.S3FileSystem(anon=True)
    annee, jour, heure = date.strftime('%Y'), date.strftime('%j'), date.strftime('%H')
    save_dir = os.path.join(GOES_DIR, zone_name, date.strftime('%Y-%m-%d_%Hh%M'))
    os.makedirs(save_dir, exist_ok=True)
    files_s3 = fs.glob(f'noaa-goes16/ABI-L1b-RadF/{annee}/{jour}/{heure}/*')
    local_files = []
    to_download = []
    for remote in files_s3:
        if any(f'M6{b}' in remote for b in BANDS) and date.strftime('s%Y%j%H00') in remote:
            local = os.path.join(save_dir, remote.split('/')[-1])
            local_files.append(local)
            if not os.path.exists(local): to_download.append(remote)
    if to_download:
        print(f"   📥 Téléchargement S3 ({len(to_download)} fichiers)...")
        fs.get(to_download, [os.path.join(save_dir, r.split('/')[-1]) for r in to_download])
    scn = Scene(filenames=local_files, reader='abi_l1b')
    scn.load(BANDS)
    return scn

def build_X_y(scn):
    masks = get_expert_masks(scn)
    shape = scn['C01'].shape
    X = np.empty((shape[0], shape[1], 12), dtype=np.float32)
    for i, b in enumerate(BANDS):
        data = np.nan_to_num(scn[b].values.astype(np.float32))
        if i <= 5: # Reflectances
            if np.max(data) > 2.0: data /= 100.0
            data = np.clip(data, 0, 1)
        X[:,:,i] = data
    Y = np.zeros(shape, dtype=np.uint8)
    for nom, masque in masks:
        for cle, idx in CLASS_MAP.items():
            if cle in nom: Y[masque] = idx; break
    return X, Y


with tab6:
        # =====================================================================
        # --- NOUVELLE SECTION : GÉNÉRATEUR DE DONNÉES S3 AVEC 10 ZONES ---
        # =====================================================================
        from datetime import datetime, date, timedelta
        import netCDF4

        # Dictionnaire des 10 zones fascinantes couvertes par GOES-16
        # Format : [Lon_min, Lat_min, Lon_max, Lat_max]
        ZONES_GOES16 = {
            "Ouragan Lee (Zoom Atlantique)": [-75, 20, -55, 35],
            "Côte Est USA": [-85, 30, -60, 50],
            "Floride & Bahamas": [-85, 22, -75, 32],
            "Golfe du Mexique": [-98, 18, -80, 32],
            "Mer des Caraïbes": [-85, 10, -60, 25],
            "Nord-Est USA & Québec": [-80, 40, -60, 55],
            "Tornado Alley (Plaines USA)": [-105, 25, -90, 40],
            "Océan Atlantique Nord": [-60, 30, -40, 50],
            "Les Grands Lacs": [-92, 40, -75, 50],
            "Amazonie Nord (Brésil/Venezuela)": [-70, -10, -50, 10]
        }

        with st.expander("☁️ Générer de nouvelles données depuis AWS S3 (Non fonctionnelle)", expanded=False):
            st.markdown("Choisissez une date et une région pour télécharger les données GOES-16 et générer vos propres fichiers `X_full.npy` et `Y_full.npy`.")
            
            # Limites de dates (1er Janvier 2020 -> Hier)
            min_date = date(2020, 1, 1)
            max_date = date.today() - timedelta(days=1)
            
            d_col1, d_col2 = st.columns([1.5, 1])
            with d_col1:
                col_date, col_zone = st.columns(2)
                with col_date:
                    user_date = st.date_input(
                        "📅 Date :",
                        value=date(2023, 9, 10), # Ouragan Lee par défaut
                        min_value=min_date,
                        max_value=max_date
                    )
                with col_zone:
                    nom_zone_choisie = st.selectbox(
                        "🗺️ Zone géographique :",
                        options=list(ZONES_GOES16.keys())
                    )
            
            with d_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Lancer le téléchargement AWS S3", type="primary", use_container_width=True):
                    
                    ZONE_COORDS = ZONES_GOES16[nom_zone_choisie]
                    # On crée un nom de dossier propre sans emojis et sans espaces
                    clean_zone_name = "".join([c if c.isalnum() else "_" for c in nom_zone_choisie.split(" ")[1]])
                    ZONE_NAME   = f'Custom_{clean_zone_name}_{user_date.strftime("%Y%m%d")}'
                    
                    RESOLUTION  = 4000
                    HEURES      = [13, 15, 17, 19, 21] 
                    dates_diurnes = [datetime(user_date.year, user_date.month, user_date.day, h, 0) for h in HEURES]
                    
                    area_def = get_area_definition(ZONE_COORDS, resolution=RESOLUTION)
                    X_list, Y_list = [], []
                    
                    progress_bar = st.progress(0, text="Initialisation de la connexion S3...")
                    status_text = st.empty()
                    
                    try:
                        for idx, dt in enumerate(dates_diurnes):
                            status_text.text(f"⏳ Téléchargement et traitement pour {dt.strftime('%H:%M')} ({idx+1}/5)...")
                            
                            scn = load_sat_data(dt, ZONE_NAME)
                            local_scn = scn.resample(area_def)
                            X, Y = build_X_y(local_scn)
                            
                            X_list.append(X.astype(np.float16)) 
                            Y_list.append(Y)
                            
                            del scn, local_scn
                            import gc; gc.collect()
                            
                            progress_bar.progress((idx + 1) / 5, text=f"Heure {dt.strftime('%H:%M')} terminée ✓")
                            
                        if len(X_list) == 5:
                            status_text.text("Empilement final et sauvegarde sur le disque...")
                            X_full = np.stack(X_list, axis=0)
                            Y_full = np.stack(Y_list, axis=0)
                            
                            file_x = f"X_custom_{user_date.strftime('%Y%m%d')}.npy"
                            file_y = f"Y_custom_{user_date.strftime('%Y%m%d')}.npy"
                            
                            np.save(file_x, X_full)
                            np.save(file_y, Y_full)
                            
                            taille_mo = os.path.getsize(file_x) / (1024*1024)
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.success(f"Fichiers générés avec succès ! ({taille_mo:.1f} MB)")
                            st.info(f"Sauvegardés sous : `{file_x}` et `{file_y}`. **Glissez-les dans la zone juste en dessous !** 👇")
                        else:
                            st.error(f"⚠️ Échec : {len(X_list)}/5 frames. Données manquantes sur AWS S3 pour cette date.")
                            
                    except Exception as e:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"❌ Erreur lors du téléchargement : {e}")
        
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            up_X = st.file_uploader("Charger X_full.npy  (T, H, W, 12)", type="npy")
        with c2:
            up_Y = st.file_uploader("Charger Y_full.npy  (T, H, W)", type="npy")

        if st.button("Lancer l'Analyse Comparative", type="primary"):
            if up_X and up_Y:
                try:
                    # ── 1. CHARGEMENT ────────────────────────────────────────────
                    X_raw    = np.load(up_X).astype(np.float32)
                    Y_expert = np.load(up_Y)

                    if X_raw.ndim == 3:
                        X_raw    = X_raw[np.newaxis]
                        Y_expert = Y_expert[np.newaxis]

                    T, H, W, C = X_raw.shape
                    st.info(f"Données chargées : {T} frames · {H}×{W} px · {C} bandes")

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # ── 2. CHARGEMENT DES MODÈLES ────────────────────────────────
                    m1 = UNet(in_channels=3).to(device)
                    m1.load_state_dict(torch.load("models/unet_best.pth",    map_location=device))
                    m2 = UNet(in_channels=4).to(device)
                    m2.load_state_dict(torch.load("models/unet_v2_best.pth", map_location=device))
                    m1.eval(); m2.eval()

                    # ── 3. TRAITEMENT PHYSIQUE ────────────────────────────────────
                    X_phys = np.empty_like(X_raw, dtype=np.float32)
                    for i in range(12):
                        data = np.nan_to_num(X_raw[..., i])
                        if i <= 5:
                            if data.max() > 2.0:
                                data = data / 100.0
                            data = np.clip(data, 0, 1)
                        X_phys[..., i] = data

                    # ── 4. NORMALISATION ─────────────────────────────────────────
                    vmin3 = np.load("models/vmin_3b.npy")
                    vmax3 = np.load("models/vmax_3b.npy")
                    vmin4 = np.load("models/vmin_4b.npy")
                    vmax4 = np.load("models/vmax_4b.npy")
                    
                    INDICES_3B = [10, 6, 3]       # C13, C07, C04
                    INDICES_4B = [10, 6, 3, 1]    # C13, C07, C04, C02

                    st.success(
                        f"**Conversion des canaux terminée :**\n"
                        f"- Extraction V1 : {len(INDICES_3B)} bandes filtrées depuis les 12 initiales (indices {INDICES_3B}).\n"
                        f"- Extraction V2 : {len(INDICES_4B)} bandes filtrées depuis les 12 initiales (indices {INDICES_4B})."
                    )

                    def normalize(img, indices, vmin, vmax):
                        x   = img[..., indices].astype(np.float32)
                        rng = vmax - vmin
                        rng[rng == 0] = 1.0
                        return np.clip((x - vmin) / rng, 0, 1)

                    def infer_tiling(model, img_hwc):
                        y_pred   = np.zeros((H, W), dtype=np.uint8)
                        y_starts = list(range(0, H - PATCH_SIZE + 1, PATCH_SIZE))
                        if y_starts and y_starts[-1] + PATCH_SIZE < H:
                            y_starts.append(H - PATCH_SIZE)
                        if not y_starts:
                            y_starts = [0]
                        x_starts = list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE))
                        if x_starts and x_starts[-1] + PATCH_SIZE < W:
                            x_starts.append(W - PATCH_SIZE)
                        if not x_starts:
                            x_starts = [0]
                        with torch.no_grad():
                            for i in y_starts:
                                for j in x_starts:
                                    patch = img_hwc[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
                                    t_in  = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).to(device)
                                    pred  = model(t_in).argmax(dim=1).squeeze().cpu().numpy()
                                    y_pred[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = pred
                        return y_pred

                    # ── 5. BOUCLE VIDÉO ───────────────────────────────────────────
                    tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp_mp4.close()
                    writer = imageio.get_writer(
                        tmp_mp4.name, fps=2, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"]
                    )

                    resultats_frames = []

                    bar    = st.progress(0, text="Inférence en cours…")
                    status = st.empty()

                    for t in range(T):
                        status.text(f"Frame {t+1}/{T} | Heure {t+1}/5 du cycle diurne")
                        img_12b = X_phys[t]

                        x3b = normalize(img_12b, INDICES_3B, vmin3, vmax3)
                        x4b = normalize(img_12b, INDICES_4B, vmin4, vmax4)
                        
                        p1  = infer_tiling(m1, x3b)
                        p2  = infer_tiling(m2, x4b)

                        resultats_frames.append({
                            "rgb":    build_rgb(img_12b),
                            "expert": Y_expert[t],
                            "p1":     p1,
                            "p2":     p2,
                        })

                        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                        fig.patch.set_facecolor("#0E1117")
                        fig.suptitle(f"Heure {t+1}/5  ·  Cycle diurne",
                                     color="white", fontsize=13, fontweight="bold")

                        axes[0].imshow(build_rgb(img_12b))
                        axes[0].set_title("RGB Synthétique",         color="white", fontsize=10)
                        axes[1].imshow(Y_expert[t], cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                        axes[1].set_title("Vérité Terrain (Expert)", color="white", fontsize=10)
                        axes[2].imshow(p1, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                        axes[2].set_title("U-Net V1 (3B)",           color="white", fontsize=10)
                        axes[3].imshow(p2, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                        axes[3].set_title("U-Net V2 (4B)",           color="white", fontsize=10)

                        for ax in axes:
                            ax.axis("off")
                            ax.set_facecolor("#0E1117")

                        fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=8,
                                   fontsize=8, facecolor="#1E1E2E", labelcolor="white",
                                   bbox_to_anchor=(0.5, -0.04))
                        plt.tight_layout()

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=100,
                                    bbox_inches="tight", facecolor=fig.get_facecolor())
                        buf.seek(0)
                        frame_arr = imageio.imread(buf)[..., :3]
                        plt.close(fig)
                        writer.append_data(frame_arr)

                        bar.progress((t + 1) / T, text=f"Frame {t+1}/{T} encodée ✓")

                    writer.close()
                    bar.empty()
                    status.empty()

                    # ── 6. SAUVEGARDE EN SESSION STATE ───────────────────────────
                    # On sauvegarde la vidéo encodée en base64 pour qu'elle survive aux clics !
                    video_b64 = base64.b64encode(open(tmp_mp4.name, "rb").read()).decode()
                    st.session_state["video_b64"] = video_b64
                    
                    st.session_state["resultats_frames"] = resultats_frames
                    st.session_state["T_frames"]         = T
                    st.session_state["H_frames"]         = H
                    st.session_state["W_frames"]         = W

                    os.unlink(tmp_mp4.name)

                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())

            else:
                st.warning("⚠️ Veuillez charger les DEUX fichiers .npy.")


        # ── SECTION RÉSULTATS (VIDÉO + PATCHES) ──────────────────────────────
        # Visible dès que les résultats existent dans le session_state
        if "resultats_frames" in st.session_state:

            # --- AFFICHAGE DE LA VIDÉO (Maintenant persistante !) ---
            st.success("Vidéo du cycle diurne disponible !")
            st.markdown(f"""
                <video width="100%" autoplay loop muted playsinline controls>
                    <source src="data:video/mp4;base64,{st.session_state['video_b64']}" type="video/mp4">
                </video>
            """, unsafe_allow_html=True)

            # --- AFFICHAGE DES PATCHES ---
            resultats_frames = st.session_state["resultats_frames"]
            T_s = st.session_state["T_frames"]
            H_s = st.session_state["H_frames"]
            W_s = st.session_state["W_frames"]

            st.markdown("---")
            st.markdown("### 🔍 Exploration par Patches Aléatoires")
            st.markdown(
                "Sélectionnez une heure du cycle diurne et générez des patches "
                "aléatoires pour comparer RGB · Expert · M1 · M2 à l'échelle 128×128."
            )

            ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
            with ctrl1:
                heure_sel = st.selectbox(
                    "Heure du cycle diurne",
                    options=list(range(T_s)),
                    format_func=lambda i: f"Heure {i+1}/5",
                    key="patch_heure"
                )
            with ctrl2:
                n_patches = st.slider(
                    "Nombre de patches", min_value=1, max_value=6, value=3,
                    key="patch_n"
                )
            with ctrl3:
                st.markdown("<br>", unsafe_allow_html=True)   # alignement vertical
                generer = st.button("Générer", key="btn_patches")

            if generer:
                frame = resultats_frames[heure_sel]
                rgb_full    = frame["rgb"]      # (H, W, 3)
                expert_full = frame["expert"]   # (H, W)
                p1_full     = frame["p1"]       # (H, W)
                p2_full     = frame["p2"]       # (H, W)

                max_i = H_s - PATCH_SIZE
                max_j = W_s - PATCH_SIZE

                patches_valides = []
                tentatives      = 0
                max_tentatives  = 200

                while len(patches_valides) < n_patches and tentatives < max_tentatives:
                    tentatives += 1
                    i = np.random.randint(0, max_i + 1)
                    j = np.random.randint(0, max_j + 1)

                    patch_expert = expert_full[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

                    if patch_expert.mean() > 0:
                        patches_valides.append((i, j))

                if not patches_valides:
                    st.warning("Aucun patch valide trouvé | la zone est peut-être vide.")
                else:
                    st.markdown(
                        f"**{len(patches_valides)} patch(es) aléatoire(s)** "
                        f"| Heure {heure_sel+1}/5 "
                        f"| taille {PATCH_SIZE}×{PATCH_SIZE} px"
                    )

                    for k, (i, j) in enumerate(patches_valides):

                        p_rgb    = rgb_full   [i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        p_expert = expert_full[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        p_pred1  = p1_full    [i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        p_pred2  = p2_full    [i:i+PATCH_SIZE, j:j+PATCH_SIZE]

                        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                        fig.patch.set_facecolor("#0E1117")
                        fig.suptitle(
                            f"Patch {k+1}  ·  position ({i}, {j})  ·  Heure {heure_sel+1}/5",
                            color="white", fontsize=11, fontweight="bold"
                        )

                        axes[0].imshow(p_rgb)
                        axes[0].set_title("RGB Synthétique",         color="white", fontsize=10)

                        axes[1].imshow(p_expert, cmap=CMAP_EXPERT, norm=NORM_EXPERT,
                                       interpolation="nearest")
                        axes[1].set_title("Vérité Terrain (Expert)", color="white", fontsize=10)

                        axes[2].imshow(p_pred1, cmap=CMAP_EXPERT, norm=NORM_EXPERT,
                                       interpolation="nearest")
                        axes[2].set_title("U-Net V1 (3B)",           color="white", fontsize=10)

                        axes[3].imshow(p_pred2, cmap=CMAP_EXPERT, norm=NORM_EXPERT,
                                       interpolation="nearest")
                        axes[3].set_title("U-Net V2 (4B)",           color="white", fontsize=10)

                        for ax in axes:
                            ax.axis("off")
                            ax.set_facecolor("#0E1117")

                        fig.legend(
                            handles=LEGEND_PATCHES, loc="lower center", ncol=8,
                            fontsize=8, facecolor="#1E1E2E", labelcolor="white",
                            bbox_to_anchor=(0.5, -0.06)
                        )
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)