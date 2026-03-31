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
from datetime import datetime, date, timedelta
import netCDF4
import s3fs
from satpy import Scene
from pyresample import create_area_def
from pyproj import CRS, Transformer

# ==========================================
# SYSTÈME BILINGUE ET EN-TÊTE (ALIGNÉS)
# ==========================================
# 1. Initialisation de la langue et de l'état du globe
if "lang" not in st.session_state:
    st.session_state.lang = "Default" # État initial pour afficher le globe

# Définition de l'icône : Globe si jamais cliqué, sinon le drapeau actif
if st.session_state.lang == "Default":
    current_icon = "🌍"
    lang = "Français" # On traite les textes en FR par défaut
else:
    lang = st.session_state.lang
    current_icon = "🇫🇷" if lang == "Français" else "🇬🇧"

def t(fr, en):
    """Traduction instantanée"""
    return fr if lang == "Français" else en

# --- CSS pour forcer la largeur du menu à celle du bouton ---
st.markdown("""
    <style>
    [data-testid="stPopoverBody"] {
        width: 80px !important;
        min-width: 80px !important;
        padding: 5px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Ligne supérieure
col_lang, col_logos, col_vide = st.columns([0.5, 4, 0.5])

with col_lang:
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    
    # Le bouton affiche le globe au début, puis le drapeau choisi
    with st.popover(current_icon):
        if st.button("🇫🇷", key="btn_fr", use_container_width=True):
            st.session_state.lang = "Français"
            st.rerun()
            
        if st.button("🇬🇧", key="btn_en", use_container_width=True):
            st.session_state.lang = "English"
            st.rerun()

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
    H, W, _ = img_12b.shape
    X_phys = np.zeros_like(img_12b, dtype=np.float32)
    for i in range(12):
        data = np.nan_to_num(img_12b[..., i])
        if i <= 5: 
            if np.max(data) > 2.0: data /= 100.0
            data = np.clip(data, 0, 1)
        X_phys[..., i] = data
        
    X_sel = X_phys[..., indices_sel]
    X_norm = np.clip((X_sel - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    y_pred = np.zeros((H, W), dtype=np.uint8)
    for i in range(0, H, PATCH_SIZE):
        for j in range(0, W, PATCH_SIZE):
            i_end, j_end = min(i + PATCH_SIZE, H), min(j + PATCH_SIZE, W)
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE, len(indices_sel)), dtype=np.float32)
            actual_p = X_norm[i:i_end, j:j_end, :]
            patch[:actual_p.shape[0], :actual_p.shape[1], :] = actual_p
            
            t_in = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(t_in).argmax(dim=1).squeeze().cpu().numpy()
            y_pred[i:i_end, j:j_end] = pred[:i_end-i, :j_end-j]
            
    return y_pred

def fig_to_array(fig):
    fig.canvas.draw()
    buf = io.BytesIO()
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
    logos_html = f"<p style='text-align:center; color:red;'>Logos introuvables / not found : {e}</p>"

html_header = f"""
{logos_html}
<div style="text-align: center; margin-top: 15px;">
    <h1 style="margin-bottom: 0px;">{t("Application de l'IA à l'Imagerie Satellitaire", "Applying AI to Satellite Imagery")}</h1>
    <h3 style="margin-top: 5px; color: #555555;">{t("Projet de 2ème Année | Approche Experte et Réduction de Dimension", "2nd Year Project | Expert Approach & Dimensionality Reduction")}</h3>
    <br>
    <p style="font-size: 18px; margin-bottom: 0px;">
        <strong>Victor Fourel</strong> &amp; <strong>Sacha Chérifi</strong>
    </p>
    <p style="font-size: 14px; margin-top: 0px;">
        <i>École Nationale Supérieure de l'Électronique et de ses Applications (ENSEA)</i>
    </p>
    <br>
    <p style="font-size: 16px; margin-bottom: 0px;">
        <strong>{t("Encadrant", "Supervisor")} ENSEA :</strong> M. Nicolas Simond<br>
        <strong>{t("Encadrants", "Supervisors")} IGN :</strong> M. Marc Poupée {t("et", "and")} M. Arnaud Le Bris
    </p>
    <br>
    <p style="font-size: 14px; color: #888888;">{t("Année 2026", "Year 2026")}</p>
</div>
<hr>
"""

st.markdown(html_header, unsafe_allow_html=True)

def afficher_section_centree(titre, image_path, texte_analyse, type_message="info"):
    st.markdown(f"<h3 style='text-align: center;'>{titre}</h3>", unsafe_allow_html=True)
    try:
        st.image(f"images/{image_path}", use_container_width=True)
    except Exception as e:
        st.error(t(f"Image introuvable : images/{image_path}", f"Image not found: images/{image_path}"))
        
    if type_message == "info": st.info(texte_analyse)
    elif type_message == "warning": st.warning(texte_analyse)
    elif type_message == "success": st.success(texte_analyse)
    st.markdown("---")

# ==========================================
# CRÉATION DES ONGLETS
# ==========================================
tab_names = [
    t("Introduction", "Introduction"),
    t("Approche Experte & Dataset", "Expert Approach & Dataset"),
    t("ACP & Sélection", "PCA & Selection"),
    t("Modèle 1 3B", "Model 1 3B"),
    t("Modèle 2 4B", "Model 2 4B"),
    t("Test en direct (Inférence)", "Live Test (Inference)")
]
tab0, tab2_expert, tab3_acp, tab4, tab5, tab6 = st.tabs(tab_names)

# ------------------------------------------
# ONGLET 0 : INTRODUCTION
# ------------------------------------------
with tab0:
    st.header(t("Introduction & Problématique", "Introduction & Problem Statement"))
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color:#1e3a5f; padding: 20px; border-radius: 10px; border-left: 5px solid #4a9eda;">
        <h4 style="color:#4a9eda; margin-top:0;">{t("Problématique", "Problem Statement")}</h4>
        <p style="font-size:16px; color:white;">
            {t("Comment reproduire un arbre de décision météorologique complexe à partir d'images satellites géostationnaires en utilisant des réseaux de neurones allégés ?", 
               "How can we replicate a complex meteorological decision tree from geostationary satellite images using lightweight neural networks?")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_sat, col_disk = st.columns(2)
    with col_sat:
        st.markdown(f"<h4 style='text-align:center;'>{t('Satellite GOES-16 (NOAA)', 'GOES-16 Satellite (NOAA)')}</h4>", unsafe_allow_html=True)
        try: st.image("images/satellite_goes16.png", use_container_width=True)
        except: st.info("Image satellite_goes16.png non trouvée")

    with col_disk:
        st.markdown(f"<h4 style='text-align:center;'>{t('Vue Disque Entier (RGB Synthétique)', 'Full Disk View (Synthetic RGB)')}</h4>", unsafe_allow_html=True)
        try: st.image("images/full_disk.jpg", use_container_width=True)
        except: st.info("Image full_disk.jpg non trouvée")

    st.markdown("---")

    st.markdown(f"""
    <h4>🛰️ {t("Le Satellite GOES-16", "The GOES-16 Satellite")}</h4>
    <ul style="font-size:15px; line-height:2;">
        <li>{t("Orbite <strong>géostationnaire</strong> | observation continue de l'Amérique", "<strong>Geostationary</strong> orbit | continuous observation of the Americas")}</li>
        <li>{t("Capteur <strong>ABI (Advanced Baseline Imager)</strong> : 16 bandes spectrales du visible à l'infrarouge thermique", "<strong>ABI (Advanced Baseline Imager)</strong> sensor: 16 spectral bands from visible to thermal infrared")}</li>
        <li>{t("Images toutes les <strong>10 minutes</strong> sur le disque entier", "Images every <strong>10 minutes</strong> for the full disk")}</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"<h4 style='text-align:center;'>{t('Pipeline de Traitement', 'Processing Pipeline')}</h4>", unsafe_allow_html=True)
    try: st.image("images/schema_pipeline_final_cut.jpg", use_container_width=True)
    except: st.info("Image schema_pipeline_final_cut.jpg non trouvée")

    st.markdown("---")

    st.markdown(f"<h4 style='text-align:center;'>{t('Reprojection Géométrique', 'Geometric Reprojection')}</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style="font-size:15px; line-height:2;">
        <li>{t("Le satellite observe la Terre comme un <strong>disque 3D déformé</strong> sur les bords", "The satellite observes Earth as a <strong>3D disk distorted</strong> at the edges")}</li>
        <li>{t("Les pixels bruts sont projetés sur une <strong>grille géographique 2D régulière</strong> via rééchantillonnage", "Raw pixels are projected onto a <strong>regular 2D geographic grid</strong> via resampling")}</li>
    </ul>
    """, unsafe_allow_html=True)
    try: 
        st.image("images/schema_data_final.jpg", use_container_width=True)
        st.caption(t("Transformation spatiale et rééchantillonnage des pixels", "Spatial transformation and pixel resampling"))
    except: st.info("Image schema_data_final.jpg non trouvée")


# ------------------------------------------
# ONGLET 1 : APPROCHE EXPERTE
# ------------------------------------------
with tab2_expert:
    st.header(t("Approche Experte & Génération du Dataset", "Expert Approach & Dataset Generation"))
    
    st.markdown(f"<h3 style='text-align:center;'>{t('Les 8 Bandes GOES-16 Utilisées', 'The 8 GOES-16 Bands Used')}</h3>", unsafe_allow_html=True)

    df_bandes = pd.DataFrame({
        t("Bande", "Band"): ["C02", "C03", "C04", "C05", "C06", "C08", "C13", "C15"],
        t("Caractéristique", "Characteristic"): [
            t("Rouge | Épaisseur optique", "Red | Optical thickness"),
            t("Végétation | Contraste Terre/Mer", "Vegetation | Land/Sea contrast"),
            t("Proche-IR | Détection Cirrus", "Near-IR | Cirrus detection"),
            t("Proche-IR | Absorption Neige", "Near-IR | Snow absorption"),
            t("Microphysique | Taille des particules", "Microphysics | Particle size"),
            t("IR | Vapeur d'eau haute", "IR | High water vapor"),
            t("IR | Température du sommet nuageux", "IR | Cloud top temperature"),
            t("IR Sale | Sensibilité glace/humidité", "Dirty IR | Ice/moisture sensitivity")
        ]
    })
    st.table(df_bandes)

    st.markdown("---")
    st.markdown(f"<h3 style='text-align:center;'>{t('Feature Engineering : Les Indices Physiques', 'Feature Engineering: Physical Indices')}</h3>", unsafe_allow_html=True)

    col_ndsi, col_otd, col_swd = st.columns(3)
    with col_ndsi:
        st.markdown(f"""
        <div style="background:#1a3a1a; padding:15px; border-radius:8px; border-left:4px solid #4CAF50;">
            <h5 style="color:#4CAF50; margin-top:0;">NDSI</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">(C02 − C05) / (C02 + C05)</p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#4CAF50;">{t("Cible : Neige", "Target: Snow")}</strong><br>
                {t("Forte réflexion C02, forte absorption C05.", "High C02 reflection, high C05 absorption.")}<br>
                {t("NDSI > 0.4 → neige détectée", "NDSI > 0.4 → snow detected")}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_otd:
        st.markdown(f"""
        <div style="background:#1a1a3a; padding:15px; border-radius:8px; border-left:4px solid #4a9eda;">
            <h5 style="color:#4a9eda; margin-top:0;">OTD</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">C08 − C13</p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#4a9eda;">{t("Cible : Convection", "Target: Convection")}</strong><br>
                {t("Dépassement stratosphérique.", "Stratospheric overshooting.")}<br>
                {t("OTD > −5.0 → Cumulonimbus", "OTD > −5.0 → Cumulonimbus")}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_swd:
        st.markdown(f"""
        <div style="background:#3a1a1a; padding:15px; border-radius:8px; border-left:4px solid #e74c3c;">
            <h5 style="color:#e74c3c; margin-top:0;">SWD</h5>
            <p style="font-size:18px; text-align:center; color:white; margin:8px 0;">C13 − C15</p>
            <p style="color:#aaa; font-size:13px; margin:0;">
                <strong style="color:#e74c3c;">{t("Cible : Glace fine", "Target: Thin ice")}</strong><br>
                {t("Discrimine Cirrus vs nuage opaque.", "Discriminates Cirrus vs opaque cloud.")}<br>
                {t("SWD > 1.5 K → Cirrus/Cs", "SWD > 1.5 K → Cirrus/Cs")}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    # Affichage des 2 images de métriques côte à côte
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image("images/otd_storm_1.jpg", use_container_width=True)
    with col_img2:
        st.image("images/otd_storm_2.jpg", use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h3 style='text-align:center;'>{t('Arbre Décisionnel Expert | Règles Logiques', 'The Expert Decision Tree | Logical Rules')}</h3>", unsafe_allow_html=True)

    df_arbre = pd.DataFrame({
        t("Classe", "Class"): ["Cumulonimbus", "Snow/Ice", "Cirrostratus", "Cirrus", "Mid-Level", "Stratus", "Cumulus", "Surface"],
        t("Conditions Physiques Principales", "Main Physical Conditions"): [
            t("T₁₃ < 235 K  ET  OTD > −5.0", "T₁₃ < 235 K  AND  OTD > −5.0"),
            t("NDSI > 0.4  ET  T₁₃ < 273 K", "NDSI > 0.4  AND  T₁₃ < 273 K"),
            t("Glace  ET  T₁₃ < 260 K  ET  C02 > 0.20", "Ice  AND  T₁₃ < 260 K  AND  C02 > 0.20"),
            t("Glace  ET  voile semi-transparent (C02 < 0.35)", "Ice  AND  semi-transparent veil (C02 < 0.35)"),
            t("Nuage eau  ET  250 K ≤ T₁₃ < 273 K", "Water cloud  AND  250 K ≤ T₁₃ < 273 K"),
            t("Nuage eau bas (T₁₃ ≥ 273 K)  ET  C02 > 0.45", "Low water cloud (T₁₃ ≥ 273 K)  AND  C02 > 0.45"),
            t("Nuage eau bas (T₁₃ ≥ 273 K)  ET  C02 ≤ 0.45", "Low water cloud (T₁₃ ≥ 273 K)  AND  C02 ≤ 0.45"),
            t("Aucun seuil nuageux ou neigeux atteint", "No cloud or snow threshold met")
        ]
    })
    st.table(df_arbre)

    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>{t('Construction du Dataset', 'Dataset Construction')}</h3>", unsafe_allow_html=True)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(f"""
        <div style="background:#1e2a1e; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#4CAF50;">128 × 128</h2>
            <p style="color:#aaa;">{t("Taille des patches<br>(pixels)", "Patch size<br>(pixels)")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
        <div style="background:#1e1e2a; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#4a9eda;">{t("3 Zones", "3 Zones")}</h2>
            <p style="color:#aaa;">{t("Golfe du Mexique<br>Côte Est USA · Caraïbes", "Gulf of Mexico<br>US East Coast · Caribbean")}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_d3:
        st.markdown(f"""
        <div style="background:#2a1e1e; padding:15px; border-radius:8px; text-align:center;">
            <h2 style="color:#e74c3c;">70/15/15</h2>
            <p style="color:#aaa;">{t("Split<br>Train / Val / Test", "Split<br>Train / Val / Test")}</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(t(
        "**⚠️ Points de vigilance majeurs :**\n"
        "- **Gestion du vide :** Exclusion des bords spatiaux (artefacts de projection à 0 ou NaN)\n"
        "- **Déséquilibre des classes :** La classe Surface est omniprésente, contrairement aux Cumulonimbus\n"
        "- **Normalisation :** Bornes physiques absolues pour ramener IR (Kelvin) et Visible (Réflectances) sur [0, 1] sans Data Shift",
        "**⚠️ Major watch points:**\n"
        "- **Void management:** Exclusion of spatial edges (projection artifacts at 0 or NaN)\n"
        "- **Class imbalance:** The Surface class is omnipresent, unlike Cumulonimbus\n"
        "- **Normalization:** Absolute physical bounds to map IR (Kelvin) and Visible (Reflectances) to [0, 1] without Data Shift"
    ))

    st.markdown(f"<h4 style='text-align:center;'>{t('Exemples de Classification Experte', 'Examples of Expert Classification')}</h4>", unsafe_allow_html=True)
    try: st.image("images/cas_extremes_expert2_comp.png", use_container_width=True)
    except: st.info("Image cas_extremes_expert2_comp.png non trouvée")


# ------------------------------------------
# ONGLET 2 : ACP
# ------------------------------------------
with tab3_acp:
    st.header(t("Analyses Statistiques & Sélection des Bandes", "Statistical Analysis & Band Selection"))

    st.markdown(f"<h3 style='text-align:center;'>{t('Analyse en Composantes Principales (ACP)', 'Principal Component Analysis (PCA)')}</h3>", unsafe_allow_html=True)
    st.info(t(
        "**Objectif :** Identifier la redondance dans les 12 bandes et réduire l'espace des caractéristiques d'entrée du modèle.",
        "**Objective:** Identify redundancy in the 12 bands and reduce the input feature space of the model."
    ))

    df_acp = pd.DataFrame({
        t("Composante", "Component"): ["PC1", "PC2", "PC3"],
        t("Information capturée", "Captured Information"): [
            t("Dynamique thermique globale | nuages hauts vs surface", "Global thermal dynamics | high clouds vs surface"),
            t("Albédo et réflectances | épaisseur optique des nuages", "Albedo and reflectances | cloud optical thickness"),
            t("Phénomènes marginaux de transition | glace/eau", "Marginal transition phenomena | ice/water")
        ]
    })
    st.table(df_acp)

    afficher_section_centree(
        t("Scree Plot & Loadings ACP", "Scree Plot & PCA Loadings"),
        "acp_variance.png",
        t("PC1 et PC2 capturent l'essentiel de la variance. Les bandes thermiques dominent PC1, les bandes visibles dominent PC2.", 
          "PC1 and PC2 capture most of the variance. Thermal bands dominate PC1, visible bands dominate PC2.")
    )

    afficher_section_centree(
        t("Visualisation 2D de l'Espace Latent", "2D Visualization of the Latent Space"),
        "acp_classes.png",
        t("Séparabilité partielle des classes dans le plan PC1/PC2. Les classes extrêmes (Surface, Cumulonimbus) sont bien isolées.",
          "Partial separability of classes in the PC1/PC2 plane. Extreme classes (Surface, Cumulonimbus) are well isolated.")
    )

    st.markdown(f"<h3 style='text-align: center;'>{t('Séparabilité des classes (ACP 3D)', 'Class Separability (3D PCA)')}</h3>", unsafe_allow_html=True)
    try:
        with open("./images/acp_3d_figure.json", "r") as f:
            json_clean = f.read()
        fig = pio.from_json(json_clean, skip_invalid=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(t(f"Erreur lors du chargement de l'ACP 3D : {e}", f"Error loading 3D PCA: {e}"))

    st.info(t("PC3 apporte une séparation supplémentaire pour les nuages de glace (Cirrus, Cirrostratus) qui se superposent sur les 2 premiers axes.", 
              "PC3 provides additional separation for ice clouds (Cirrus, Cirrostratus) that overlap on the first 2 axes."))

    st.markdown(f"<h3 style='text-align:center;'>{t('Matrice de Corrélation de Pearson', 'Pearson Correlation Matrix')}</h3>", unsafe_allow_html=True)
    afficher_section_centree(
        t("Corrélations Inter-Bandes", "Inter-Band Correlations"),
        "correlation_bandes.png",
        t("Démarcation franche entre deux familles : C01→C06 (Visible/Proche-IR, très corrélées) et C07→C15 (Infrarouge Thermique). Éviter les paires redondantes en entrée du modèle.",
          "Clear demarcation between two families: C01→C06 (Visible/Near-IR, highly correlated) and C07→C15 (Thermal Infrared). Avoid redundant pairs as model input.")
    )

    st.markdown(f"<h3 style='text-align:center;'>{t('Pouvoir Discriminant : Score F-ANOVA', 'Discriminant Power: F-ANOVA Score')}</h3>", unsafe_allow_html=True)
    afficher_section_centree(
        t("Score F-ANOVA par Bande", "F-ANOVA Score by Band"),
        "score_F_bandes.png",
        t("Le Score F mesure le rapport variance inter-classes / variance intra-classes. Plus F est élevé, mieux la bande discrimine les 8 classes nuageuses.",
          "The F-Score measures the ratio of inter-class variance / intra-class variance. The higher the F, the better the band discriminates the 8 cloud classes.")
    )

    st.markdown(f"<h3 style='text-align:center;'>{t('Synthèse : Sélection des Bandes Optimales', 'Summary: Optimal Band Selection')}</h3>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown(f"""
        <div style="background:#1e2a3a; padding:20px; border-radius:8px; border-top:4px solid #4a9eda; text-align:center;">
            <h3 style="color:#4a9eda;">C13</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">{t("Infrarouge Thermique", "Thermal Infrared")}</strong><br>
                {t("Majorité de la variance (PC1)<br>→ Altitude des sommets nuageux", "Majority of variance (PC1)<br>→ Cloud top altitude")}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b2:
        st.markdown(f"""
        <div style="background:#2a1e3a; padding:20px; border-radius:8px; border-top:4px solid #9b59b6; text-align:center;">
            <h3 style="color:#9b59b6;">C07</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">{t("Ondes Courtes IR", "Shortwave IR")}</strong><br>
                {t("Proxy statistique brouillards/stratus<br>→ Absente de l'arbre expert !", "Statistical proxy for fog/stratus<br>→ Absent from the expert tree!")}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b3:
        st.markdown(f"""
        <div style="background:#1a3a2a; padding:20px; border-radius:8px; border-top:4px solid #4CAF50; text-align:center;">
            <h3 style="color:#4CAF50;">C04</h3>
            <p style="color:#aaa; font-size:13px;">
                <strong style="color:white;">{t("Proche Infrarouge", "Near Infrared")}</strong><br>
                {t("Information orthogonale (PC2)<br>→ Distinction des Cirrus fins", "Orthogonal information (PC2)<br>→ Distinction of thin Cirrus")}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.success(t(
        "**Le raccourci mathématique de l'IA**\nL'IA ne copie pas la logique humaine. Ces 3 bandes suffisent à retrouver l'information latente exploitée par l'arbre expert à 8 bandes | avec en plus C02 pour le Modèle 2 (+28.5% mIoU).",
        "**The AI's mathematical shortcut**\nAI does not copy human logic. These 3 bands are enough to recover the latent information exploited by the 8-band expert tree | plus C02 for Model 2 (+28.5% mIoU)."
    ))

# ------------------------------------------
# ONGLET MODÈLE 1
# ------------------------------------------

with tab4:
    st.header(t("Analyse du Modèle 1 (3 Bandes : C13, C07, C04 | 30 Epochs)", "Analysis of Model 1 (3 Bands: C13, C07, C04 | 30 Epochs)"))
    
    # Affichage des 2 images de métriques côte à côte
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image("images/miou.webp", use_container_width=True, caption=t("Métriques utilisées (1)", "Used metrics (1)"))
    with col_img2:
        st.image("images/dice.jpg", use_container_width=True, caption=t("Métriques utilisées (2)", "Used metrics (2)"))
    st.markdown("---")

    afficher_section_centree(
        t("Courbes d'entraînement", "Training Curves"),
        "eval_4_training_curves.png",
        t("La Train Loss descend régulièrement de 1.35 à 0.60, signe d'un apprentissage stable. La Val Loss converge vers 0.52 sans diverger | pas d'overfitting. En revanche le mIoU Val oscille fortement (0.38 → 0.55) sans jamais se stabiliser, révélant que le modèle n'a pas encore trouvé de frontières de décision stables entre classes proches.",
          "The Train Loss steadily decreases from 1.35 to 0.60, a sign of stable learning. The Val Loss converges towards 0.52 without diverging | no overfitting. However, the Val mIoU oscillates strongly (0.38 → 0.55) without ever stabilizing, revealing that the model has not yet found stable decision boundaries between similar classes.")
    )

    afficher_section_centree(
        t("Matrice de Confusion", "Confusion Matrix"),
        "eval_1_confusion_matrix.png",
        t("Deux points forts nets : Surface (0.86) et Cumulonimbus (0.90). Le point faible majeur est la confusion Stratus/Cumulus : 32% des Stratus sont classés Cumulus et 18% des Cumulus sont classés Stratus | ces deux classes ont des signatures thermiques quasi-identiques sans la bande visible C02.",
          "Two clear strengths: Surface (0.86) and Cumulonimbus (0.90). The major weakness is the Stratus/Cumulus confusion: 32% of Stratus are classified as Cumulus and 18% of Cumulus are classified as Stratus | these two classes have nearly identical thermal signatures without the visible band C02.")
    )

    afficher_section_centree(
        t("Distribution de classe", "Class Distribution"),
        "eval_5_distribution.png",
        t("Biais de sous-estimation de la Surface compensé par une surestimation du Cumulus et du Stratus. Le modèle sur-segmente les nuages bas au détriment de la surface dégagée.",
          "Underestimation bias for Surface offset by an overestimation of Cumulus and Stratus. The model over-segments low clouds at the expense of clear surface."),
        type_message="warning"
    )

    afficher_section_centree(
        t("Radar de performance", "Performance Radar"),
        "eval_6_radar.png",
        t("Profil en 'comète' très caractéristique : deux pointes vers Surface et Cumulonimbus, puis un effondrement brutal sur Cumulus et Stratus.",
          "Very characteristic 'comet' profile: two points towards Surface and Cumulonimbus, then a brutal collapse on Cumulus and Stratus."),
        type_message="warning"
    )

    afficher_section_centree(
        t("Métriques par classe", "Metrics by Class"),
        "eval_2_iou_f1_precision_recall.png",
        t("mIoU global de 0.588 et Macro-F1 de 0.720. Écart de performance extrême entre Surface et Stratus.",
          "Global mIoU of 0.588 and Macro-F1 of 0.720. Extreme performance gap between Surface and Stratus."),
        type_message="warning"
    )

    afficher_section_centree(
        "Dice Scores",
        "eval_3_dice.png",
        t("La heatmap confirme le diagnostic : les nuages bas représentent le plafond de performance du Modèle 1 | sans bande visible, l'épaisseur optique est invisible.",
          "The heatmap confirms the diagnosis: low clouds represent the performance ceiling of Model 1 | without a visible band, optical thickness is invisible."),
        type_message="warning"
    )

    afficher_section_centree(
        t("Prédictions vs Réel (Modèle 1)", "Predictions vs Actual (Model 1)"),
        "PredictionvsterrainM1.png",
        t("Visuellement, la segmentation spatiale est cohérente sur les grandes structures. Les erreurs se concentrent aux frontières Cumulus/Stratus.",
          "Visually, the spatial segmentation is coherent on large structures. Errors are concentrated at the Cumulus/Stratus boundaries.")
    )

    st.markdown(f"<h3 style='text-align: center;'>{t('Comparaison Dynamique | Timelapse Modèle 1', 'Dynamic Comparison | Model 1 Timelapse')}</h3>", unsafe_allow_html=True)
    autoplay_video("images/timelapse_v1_fusionne.mp4")
    st.info(t("Sur le timelapse, on observe la montée en puissance des Cumulonimbus l'après-midi (13h→19h UTC).", 
              "In the timelapse, we observe the rise in power of Cumulonimbus in the afternoon (13h→19h UTC)."))
    st.warning(t("**Résumé Modèle 1 :** Base solide sur les phénomènes à fort contraste thermique. Le plafond à mIoU=0.588 est structurel.",
                 "**Model 1 Summary:** Solid foundation on high thermal contrast phenomena. The ceiling at mIoU=0.588 is structural."))
    
# ------------------------------------------
# ONGLET MODÈLE 2
# ------------------------------------------
with tab5:
    st.header(t("Analyse du Modèle 2 (4 Bandes : C13, C07, C04, C02 | 30 Epochs)", "Analysis of Model 2 (4 Bands: C13, C07, C04, C02 | 30 Epochs)"))
    
    afficher_section_centree(
        t("Courbes d'entraînement", "Training Curves"),
        "eval_4_training_curvesM2.png",
        t("Saut qualitatif immédiat dès l'epoch 1 : réduction de 48% de la Val Loss vs le Modèle 1. Preuve que C02 fournit un signal discriminant fort et stable.",
          "Immediate qualitative leap from epoch 1: a 48% reduction in Val Loss vs Model 1. Proof that C02 provides a strong and stable discriminant signal."),
        type_message="success"
    )

    afficher_section_centree(
        t("Matrice de Confusion", "Confusion Matrix"),
        "eval_1_confusion_matrixM2.png",
        t("Transformation radicale. Stratus passe de 0.60 à 0.98 de rappel | quasi-disparition des erreurs.",
          "Radical transformation. Stratus recall goes from 0.60 to 0.98 | near-disappearance of errors."),
        type_message="success"
    )

    afficher_section_centree(
        t("Distribution de classe", "Class Distribution"),
        "eval_5_distributionM2.png",
        t("Alignement quasi parfait entre distributions réelle et prédite pour toutes les classes.",
          "Near-perfect alignment between actual and predicted distributions for all classes."),
        type_message="success"
    )

    afficher_section_centree(
        t("Radar de performance", "Performance Radar"),
        "eval_6_radarM2.png",
        t("Transformation du profil 'comète' en profil quasi-circulaire. Toutes les classes sont désormais entre IoU=0.62 et IoU=0.94.",
          "Transformation of the 'comet' profile into a nearly circular profile. All classes are now between IoU=0.62 and IoU=0.94."),
        type_message="success"
    )

    afficher_section_centree(
        t("Métriques par classe", "Metrics by Class"),
        "eval_2_iou_f1_precision_recallM2.png",
        t("mIoU de 0.756 (+28.5%) et Macro-F1 de 0.858 (+19.1%).",
          "mIoU of 0.756 (+28.5%) and Macro-F1 of 0.858 (+19.1%)."),
        type_message="success"
    )

    afficher_section_centree(
        "Dice Scores",
        "eval_3_diceM2.png",
        t("Toutes les classes dépassent le seuil 0.76. La bande C02 a résolu structurellement la faiblesse Stratus/Cumulus.",
          "All classes exceed the 0.76 threshold. The C02 band structurally resolved the Stratus/Cumulus weakness."),
        type_message="success"
    )

    afficher_section_centree(
        t("Prédictions vs Réel (Modèle 2)", "Predictions vs Actual (Model 2)"),
        "unet_predictions_rgbM2.png",
        t("Fidélité remarquable à l'arbre expert. Les frontières entre classes sont nettes.",
          "Remarkable fidelity to the expert tree. Boundaries between classes are sharp."),
        type_message="success"
    )

    st.markdown(f"<h3 style='text-align: center;'>{t('Comparaison Dynamique | Timelapse Modèle 2', 'Dynamic Comparison | Model 2 Timelapse')}</h3>", unsafe_allow_html=True)
    autoplay_video("images/timelapse_v2_fusionne.mp4")
    st.success(t("Stabilité temporelle nettement supérieure. L'évolution du cycle convectif diurne est fidèlement reproduite.",
                 "Significantly superior temporal stability. The evolution of the diurnal convective cycle is faithfully reproduced."))
    st.success(t("**Résumé Modèle 2 :** L'ajout de C02 apporte l'information d'épaisseur optique manquante. Le réseau égale la logique experte sur 7 des 8 classes.",
                 "**Model 2 Summary:** The addition of C02 provides the missing optical thickness information. The network matches expert logic on 7 of 8 classes."))

# ------------------------------------------
# ONGLET 6 : TEST EN DIRECT (INFERENCE)
# ------------------------------------------
RESOLUTION  = 4000 
GOES_DIR    = './goes_images'
BANDS       = ['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C13','C15']
CLASS_MAP   = {'Surface':1,'Cumulus':2,'Stratus':3,'Mid-Level':4,'Cirrus':5,'Cirrostratus':6,'Snow/Ice':7,'Cumulonimbus':8}

def get_area_definition(zone_coords, resolution=4000):
    proj = {'proj': 'eqc', 'datum': 'WGS84', 'lat_ts': 0, 'lon_0': 0}
    t_proj = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_dict(proj), always_xy=True)
    xmin, ymin = t_proj.transform(zone_coords[0], zone_coords[1])
    xmax, ymax = t_proj.transform(zone_coords[2], zone_coords[3])
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

def load_sat_data(date_obj, zone_name):
    fs = s3fs.S3FileSystem(anon=True)
    annee, jour, heure = date_obj.strftime('%Y'), date_obj.strftime('%j'), date_obj.strftime('%H')
    save_dir = os.path.join(GOES_DIR, zone_name, date_obj.strftime('%Y-%m-%d_%Hh%M'))
    os.makedirs(save_dir, exist_ok=True)
    files_s3 = fs.glob(f'noaa-goes16/ABI-L1b-RadF/{annee}/{jour}/{heure}/*')
    local_files = []
    to_download = []
    for remote in files_s3:
        if any(f'M6{b}' in remote for b in BANDS) and date_obj.strftime('s%Y%j%H00') in remote:
            local = os.path.join(save_dir, remote.split('/')[-1])
            local_files.append(local)
            if not os.path.exists(local): to_download.append(remote)
    if to_download:
        print(f"   📥 AWS S3 Download ({len(to_download)} files)...")
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
        if i <= 5:
            if np.max(data) > 2.0: data /= 100.0
            data = np.clip(data, 0, 1)
        X[:,:,i] = data
    Y = np.zeros(shape, dtype=np.uint8)
    for nom, masque in masks:
        for cle, idx in CLASS_MAP.items():
            if cle in nom: Y[masque] = idx; break
    return X, Y

with tab6:
    ZONES_GOES16_FR = {
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
    
    ZONES_GOES16_EN = {
        "Hurricane Lee (Atlantic Zoom)": [-75, 20, -55, 35],
        "US East Coast": [-85, 30, -60, 50],
        "Florida & Bahamas": [-85, 22, -75, 32],
        "Gulf of Mexico": [-98, 18, -80, 32],
        "Caribbean Sea": [-85, 10, -60, 25],
        "US Northeast & Quebec": [-80, 40, -60, 55],
        "Tornado Alley (US Plains)": [-105, 25, -90, 40],
        "North Atlantic Ocean": [-60, 30, -40, 50],
        "The Great Lakes": [-92, 40, -75, 50],
        "Northern Amazonia (Brazil/Venezuela)": [-70, -10, -50, 10]
    }
    
    ZONES_GOES16 = ZONES_GOES16_FR if lang == "Français" else ZONES_GOES16_EN

    with st.expander(t("☁️ Générer de nouvelles données depuis AWS S3 (Non fonctionnelle)", "☁️ Generate new data from AWS S3 (Non-functional)"), expanded=False):
        st.markdown(t("Choisissez une date et une région pour télécharger les données GOES-16 et générer vos propres fichiers `X_full.npy` et `Y_full.npy`.", 
                      "Choose a date and a region to download GOES-16 data and generate your own `X_full.npy` and `Y_full.npy` files."))
        
        min_date = date(2020, 1, 1)
        max_date = date.today() - timedelta(days=1)
        
        d_col1, d_col2 = st.columns([1.5, 1])
        with d_col1:
            col_date, col_zone = st.columns(2)
            with col_date:
                user_date = st.date_input(
                    t("📅 Date :", "📅 Date:"),
                    value=date(2023, 9, 10),
                    min_value=min_date,
                    max_value=max_date
                )
            with col_zone:
                nom_zone_choisie = st.selectbox(
                    t("🗺️ Zone géographique :", "🗺️ Geographic Zone:"),
                    options=list(ZONES_GOES16.keys())
                )
        
        with d_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(t("Lancer le téléchargement AWS S3", "Start AWS S3 Download"), type="primary", use_container_width=True):
                ZONE_COORDS = ZONES_GOES16[nom_zone_choisie]
                clean_zone_name = "".join([c if c.isalnum() else "_" for c in nom_zone_choisie.split(" ")[1]])
                ZONE_NAME   = f'Custom_{clean_zone_name}_{user_date.strftime("%Y%m%d")}'
                
                RESOLUTION  = 4000
                HEURES      = [13, 15, 17, 19, 21] 
                dates_diurnes = [datetime(user_date.year, user_date.month, user_date.day, h, 0) for h in HEURES]
                
                area_def = get_area_definition(ZONE_COORDS, resolution=RESOLUTION)
                X_list, Y_list = [], []
                
                progress_bar = st.progress(0, text=t("Initialisation de la connexion S3...", "Initializing S3 connection..."))
                status_text = st.empty()
                
                try:
                    for idx, dt in enumerate(dates_diurnes):
                        status_text.text(t(f"⏳ Téléchargement et traitement pour {dt.strftime('%H:%M')} ({idx+1}/5)...", 
                                           f"⏳ Downloading and processing for {dt.strftime('%H:%M')} ({idx+1}/5)..."))
                        
                        scn = load_sat_data(dt, ZONE_NAME)
                        local_scn = scn.resample(area_def)
                        X, Y = build_X_y(local_scn)
                        
                        X_list.append(X.astype(np.float16)) 
                        Y_list.append(Y)
                        
                        del scn, local_scn
                        import gc; gc.collect()
                        
                        progress_bar.progress((idx + 1) / 5, text=t(f"Heure {dt.strftime('%H:%M')} terminée ✓", f"Hour {dt.strftime('%H:%M')} completed ✓"))
                        
                    if len(X_list) == 5:
                        status_text.text(t("Empilement final et sauvegarde sur le disque...", "Final stacking and saving to disk..."))
                        X_full = np.stack(X_list, axis=0)
                        Y_full = np.stack(Y_list, axis=0)
                        
                        file_x = f"X_custom_{user_date.strftime('%Y%m%d')}.npy"
                        file_y = f"Y_custom_{user_date.strftime('%Y%m%d')}.npy"
                        
                        np.save(file_x, X_full)
                        np.save(file_y, Y_full)
                        
                        taille_mo = os.path.getsize(file_x) / (1024*1024)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(t(f"Fichiers générés avec succès ! ({taille_mo:.1f} MB)", f"Files successfully generated! ({taille_mo:.1f} MB)"))
                        st.info(t(f"Sauvegardés sous : `{file_x}` et `{file_y}`. **Glissez-les dans la zone juste en dessous !** 👇", 
                                  f"Saved as: `{file_x}` and `{file_y}`. **Drag them into the area just below!** 👇"))
                    else:
                        st.error(t(f"⚠️ Échec : {len(X_list)}/5 frames. Données manquantes sur AWS S3 pour cette date.", 
                                   f"⚠️ Failure: {len(X_list)}/5 frames. Missing data on AWS S3 for this date."))
                        
                except Exception as e:
                    status_text.empty()
                    progress_bar.empty()
                    st.error(t(f"❌ Erreur lors du téléchargement : {e}", f"❌ Download error: {e}"))
    
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        up_X = st.file_uploader(t("Charger X_full.npy  (T, H, W, 12)", "Upload X_full.npy  (T, H, W, 12)"), type="npy")
    with c2:
        up_Y = st.file_uploader(t("Charger Y_full.npy  (T, H, W)", "Upload Y_full.npy  (T, H, W)"), type="npy")

    if st.button(t("Lancer l'Analyse Comparative", "Run Comparative Analysis"), type="primary"):
        if up_X and up_Y:
            try:
                X_raw    = np.load(up_X).astype(np.float32)
                Y_expert = np.load(up_Y)

                if X_raw.ndim == 3:
                    X_raw    = X_raw[np.newaxis]
                    Y_expert = Y_expert[np.newaxis]

                T, H, W, C = X_raw.shape
                st.info(t(f"Données chargées : {T} frames · {H}×{W} px · {C} bandes", f"Data loaded: {T} frames · {H}×{W} px · {C} bands"))

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                m1 = UNet(in_channels=3).to(device)
                m1.load_state_dict(torch.load("models/unet_best.pth",    map_location=device))
                m2 = UNet(in_channels=4).to(device)
                m2.load_state_dict(torch.load("models/unet_v2_best.pth", map_location=device))
                m1.eval(); m2.eval()

                X_phys = np.empty_like(X_raw, dtype=np.float32)
                for i in range(12):
                    data = np.nan_to_num(X_raw[..., i])
                    if i <= 5:
                        if data.max() > 2.0:
                            data = data / 100.0
                        data = np.clip(data, 0, 1)
                    X_phys[..., i] = data

                vmin3 = np.load("models/vmin_3b.npy")
                vmax3 = np.load("models/vmax_3b.npy")
                vmin4 = np.load("models/vmin_4b.npy")
                vmax4 = np.load("models/vmax_4b.npy")
                
                INDICES_3B = [10, 6, 3]       
                INDICES_4B = [10, 6, 3, 1]    

                st.success(t(
                    f"**Conversion des canaux terminée :**\n"
                    f"- Extraction V1 : {len(INDICES_3B)} bandes filtrées depuis les 12 initiales (indices {INDICES_3B}).\n"
                    f"- Extraction V2 : {len(INDICES_4B)} bandes filtrées depuis les 12 initiales (indices {INDICES_4B}).",
                    f"**Channel conversion completed:**\n"
                    f"- V1 Extraction: {len(INDICES_3B)} bands filtered from the initial 12 (indices {INDICES_3B}).\n"
                    f"- V2 Extraction: {len(INDICES_4B)} bands filtered from the initial 12 (indices {INDICES_4B})."
                ))

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
                    if not y_starts: y_starts = [0]
                    x_starts = list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE))
                    if x_starts and x_starts[-1] + PATCH_SIZE < W:
                        x_starts.append(W - PATCH_SIZE)
                    if not x_starts: x_starts = [0]
                    with torch.no_grad():
                        for i in y_starts:
                            for j in x_starts:
                                patch = img_hwc[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
                                t_in  = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).to(device)
                                pred  = model(t_in).argmax(dim=1).squeeze().cpu().numpy()
                                y_pred[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = pred
                    return y_pred

                tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_mp4.close()
                writer = imageio.get_writer(
                    tmp_mp4.name, fps=2, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"]
                )

                resultats_frames = []

                bar    = st.progress(0, text=t("Inférence en cours…", "Inference in progress..."))
                status = st.empty()

                for time_idx in range(T):
                    status.text(t(f"Frame {time_idx+1}/{T} | Heure {time_idx+1}/5 du cycle diurne", f"Frame {time_idx+1}/{T} | Hour {time_idx+1}/5 of the diurnal cycle"))
                    img_12b = X_phys[time_idx]

                    x3b = normalize(img_12b, INDICES_3B, vmin3, vmax3)
                    x4b = normalize(img_12b, INDICES_4B, vmin4, vmax4)
                    
                    p1  = infer_tiling(m1, x3b)
                    p2  = infer_tiling(m2, x4b)

                    resultats_frames.append({
                        "rgb":    build_rgb(img_12b),
                        "expert": Y_expert[time_idx],
                        "p1":     p1,
                        "p2":     p2,
                    })

                    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                    fig.patch.set_facecolor("#0E1117")
                    fig.suptitle(t(f"Heure {time_idx+1}/5  ·  Cycle diurne", f"Hour {time_idx+1}/5  ·  Diurnal cycle"),
                                 color="white", fontsize=13, fontweight="bold")

                    axes[0].imshow(build_rgb(img_12b))
                    axes[0].set_title(t("RGB Synthétique", "Synthetic RGB"),         color="white", fontsize=10)
                    axes[1].imshow(Y_expert[time_idx], cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[1].set_title(t("Vérité Terrain (Expert)", "Ground Truth (Expert)"), color="white", fontsize=10)
                    axes[2].imshow(p1, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[2].set_title(t("U-Net V1 (3B)", "U-Net V1 (3B)"),           color="white", fontsize=10)
                    axes[3].imshow(p2, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[3].set_title(t("U-Net V2 (4B)", "U-Net V2 (4B)"),           color="white", fontsize=10)

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

                    bar.progress((time_idx + 1) / T, text=t(f"Frame {time_idx+1}/{T} encodée ✓", f"Frame {time_idx+1}/{T} encoded ✓"))

                writer.close()
                bar.empty()
                status.empty()

                video_b64 = base64.b64encode(open(tmp_mp4.name, "rb").read()).decode()
                st.session_state["video_b64"] = video_b64
                
                st.session_state["resultats_frames"] = resultats_frames
                st.session_state["T_frames"]         = T
                st.session_state["H_frames"]         = H
                st.session_state["W_frames"]         = W

                os.unlink(tmp_mp4.name)

            except Exception as e:
                st.error(t(f"❌ Erreur : {e}", f"❌ Error: {e}"))
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(t("⚠️ Veuillez charger les DEUX fichiers .npy.", "⚠️ Please upload BOTH .npy files."))

    if "resultats_frames" in st.session_state:
        st.success(t("Vidéo du cycle diurne disponible !", "Diurnal cycle video available!"))
        st.markdown(f"""
            <video width="100%" autoplay loop muted playsinline controls>
                <source src="data:video/mp4;base64,{st.session_state['video_b64']}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)

        resultats_frames = st.session_state["resultats_frames"]
        T_s = st.session_state["T_frames"]
        H_s = st.session_state["H_frames"]
        W_s = st.session_state["W_frames"]

        st.markdown("---")
        st.markdown(f"### 🔍 {t('Exploration par Patches Aléatoires', 'Random Patches Exploration')}")
        st.markdown(t(
            "Sélectionnez une heure du cycle diurne et générez des patches aléatoires pour comparer RGB · Expert · M1 · M2 à l'échelle 128×128.",
            "Select a diurnal cycle hour and generate random patches to compare RGB · Expert · M1 · M2 at 128×128 scale."
        ))

        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
        with ctrl1:
            heure_sel = st.selectbox(
                t("Heure du cycle diurne", "Diurnal cycle hour"),
                options=list(range(T_s)),
                format_func=lambda i: t(f"Heure {i+1}/5", f"Hour {i+1}/5"),
                key="patch_heure"
            )
        with ctrl2:
            n_patches = st.slider(
                t("Nombre de patches", "Number of patches"), min_value=1, max_value=6, value=3,
                key="patch_n"
            )
        with ctrl3:
            st.markdown("<br>", unsafe_allow_html=True)
            generer = st.button(t("Générer", "Generate"), key="btn_patches")

        if generer:
            frame = resultats_frames[heure_sel]
            rgb_full    = frame["rgb"]      
            expert_full = frame["expert"]   
            p1_full     = frame["p1"]       
            p2_full     = frame["p2"]       

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
                st.warning(t("Aucun patch valide trouvé | la zone est peut-être vide.", "No valid patch found | the zone might be empty."))
            else:
                st.markdown(t(
                    f"**{len(patches_valides)} patch(es) aléatoire(s)** | Heure {heure_sel+1}/5 | taille {PATCH_SIZE}×{PATCH_SIZE} px",
                    f"**{len(patches_valides)} random patch(es)** | Hour {heure_sel+1}/5 | size {PATCH_SIZE}×{PATCH_SIZE} px"
                ))

                for k, (i, j) in enumerate(patches_valides):

                    p_rgb    = rgb_full   [i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                    p_expert = expert_full[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                    p_pred1  = p1_full    [i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                    p_pred2  = p2_full    [i:i+PATCH_SIZE, j:j+PATCH_SIZE]

                    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                    fig.patch.set_facecolor("#0E1117")
                    fig.suptitle(t(
                        f"Patch {k+1}  ·  position ({i}, {j})  ·  Heure {heure_sel+1}/5",
                        f"Patch {k+1}  ·  position ({i}, {j})  ·  Hour {heure_sel+1}/5"
                    ), color="white", fontsize=11, fontweight="bold")

                    axes[0].imshow(p_rgb)
                    axes[0].set_title(t("RGB Synthétique", "Synthetic RGB"), color="white", fontsize=10)

                    axes[1].imshow(p_expert, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[1].set_title(t("Vérité Terrain (Expert)", "Ground Truth (Expert)"), color="white", fontsize=10)

                    axes[2].imshow(p_pred1, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[2].set_title(t("U-Net V1 (3B)", "U-Net V1 (3B)"), color="white", fontsize=10)

                    axes[3].imshow(p_pred2, cmap=CMAP_EXPERT, norm=NORM_EXPERT, interpolation="nearest")
                    axes[3].set_title(t("U-Net V2 (4B)", "U-Net V2 (4B)"), color="white", fontsize=10)

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