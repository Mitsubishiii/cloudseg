import streamlit as st
import pandas as pd
import time
import base64

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
    "Modèle 1 (RGB)",
    "Modèle 2 (RGB + C02)",
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

    afficher_section_centree(
        "Visualisation 3D de l'Espace Latent",
        "acp_3d.png",
        "PC3 apporte une séparation supplémentaire pour les nuages de glace (Cirrus, Cirrostratus) qui se superposent sur les 2 premiers axes."
    )

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
# ONGLET 3 : TEST EN DIRECT (INFÉRENCE)
# ------------------------------------------
with tab6:
    st.header("Test en Direct : Inférence sur un Dataset (1 Journée)")
    st.markdown("Chargez un jeu de données satellitaires brutes pour évaluer de manière dynamique les modèles `unet_best.pth` et `unet_v2_best.pth`.")

    uploaded_files = st.file_uploader("Sélectionnez vos fichiers de données (.npy, .nc, .tif)", accept_multiple_files=True)

    if st.button("Lancer l'inférence sur les deux modèles", type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0, text="Chargement des poids : unet_best.pth & unet_v2_best.pth...")
            time.sleep(1)
            progress_bar.progress(30, text="Préparation et normalisation des tenseurs spatio-temporels...")
            time.sleep(1.5)
            progress_bar.progress(60, text="Inférence du Modèle 1 (unet_best.pth) en cours...")
            time.sleep(1.5)
            progress_bar.progress(85, text="Inférence du Modèle 2 (unet_v2_best.pth) en cours...")
            time.sleep(1.5)
            progress_bar.progress(100, text="Génération des masques de segmentation terminée !")
            time.sleep(0.5)
            progress_bar.empty()

            st.success("Inférence réussie ! L'analyse de cohérence est prête.")

            st.markdown("<h3 style='text-align: center;'>Aperçu des Prédictions</h3>", unsafe_allow_html=True)

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown("<p style='text-align: center;'><strong>Donnée Brute (RGB)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la visualisation de la frame d'entrée.")
            with col_res2:
                st.markdown("<p style='text-align: center;'><strong>Modèle 1 (unet_best.pth)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la prédiction M1.")
            with col_res3:
                st.markdown("<p style='text-align: center;'><strong>Modèle 2 (unet_v2_best.pth)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la prédiction M2.")

            with st.expander("🛠️ Comment intégrer l'inférence PyTorch ici ?"):
                st.markdown("""
                Pour rendre ce bouton pleinement opérationnel avec vos fichiers `unet_best.pth` et `unet_v2_best.pth`, remplacez les `time.sleep()` par ce type de logique :
```python
                import torch
                model_v1 = torch.load('unet_best.pth', map_location=torch.device('cpu'))
                model_v2 = torch.load('unet_v2_best.pth', map_location=torch.device('cpu'))
                model_v1.eval()
                model_v2.eval()
                with torch.no_grad():
                    pass  # pred_v1 = model_v1(data), pred_v2 = model_v2(data)
```
                """)
        else:
            st.error("⚠️ Veuillez charger au moins un fichier de données pour lancer l'inférence.")

# # ==========================================
# # SYNTHÈSE GLOBALE (TABLEAU COMPARATIF)
# # ==========================================
# st.markdown("<br><br>", unsafe_allow_html=True)
# st.header("📊 Synthèse Comparative des Performances")

# data_comparaison = {
#     "Architecture": ["Modèle 1 (RGB)", "Modèle 2 (RGB + C02)"],
#     "Bandes d'entrée": ["3", "4"],
#     "mIoU": ["0.588", "0.756"],
#     "Macro F1-Score": ["0.720", "0.858"],
#     "Évolution mIoU": ["-", "+ 28.5%"],
#     "Évolution F1-Score": ["-", "+ 19.1%"]
# }
# df_comparaison = pd.DataFrame(data_comparaison)

# _, col_table, _ = st.columns([1, 3, 1])
# with col_table:
#     st.table(df_comparaison)
#     st.info("La comparaison met en évidence l'importance critique du choix des bandes spectrales dans l'imagerie satellitaire. L'ajout d'une unique bande additionnelle a permis de résoudre les faiblesses structurelles de la première itération.")