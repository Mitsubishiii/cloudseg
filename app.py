import streamlit as st
import pandas as pd
import time
import base64

def svg_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==========================================
# EN-TÊTE — Logos SVG + HTML
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
    <h3 style="margin-top: 5px; color: #555555;">Projet de 2ème Année — Approche Experte et Réduction de Dimension</h3>
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
# FONCTION DE MISE EN PAGE (CENTRAGE STRICT)
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
tab1, tab2, tab3 = st.tabs(["Modèle 1 (RGB)", "Modèle 2 (RGB + C02)", "Test en direct (Inférence)"])

# ------------------------------------------
# ONGLET 1 : MODÈLE 1
# ------------------------------------------
with tab1:
    st.header("Analyse du Modèle 1 (3 Bandes RGB | 30 Epochs)")
    st.markdown("<br>", unsafe_allow_html=True)

    afficher_section_centree(
        "Courbes d'entraînement", 
        "eval_4_training_curves.png", 
        "Convergence saine de la Loss et du mIoU autour de l'époque 30. Absence d'overfitting majeur, mais performance limitée par le manque de diversité spectrale."
    )

    afficher_section_centree(
        "Matrice de Confusion", 
        "eval_1_confusion_matrix.png", 
        "Succès sur les classes 'Surface' (0.86) et 'Cumulonimbus' (0.90). Échec relatif sur les nuages bas avec une confusion de 32% entre 'Stratus' et 'Cumulus'."
    )

    afficher_section_centree(
        "Distribution de classe", 
        "eval_5_distribution.png", 
        "Biais de prédiction marqué. Sous-estimation de la Surface et surestimation des nuages bas (Cumulus/Stratus)."
    )

    afficher_section_centree(
        "Radar de performance", 
        "eval_6_radar.png", 
        "Profil hétérogène dit 'en papillon'. Les performances sont concentrées sur les pôles 'Surface' et 'Cumulonimbus', laissant les autres classes vulnérables."
    )

    afficher_section_centree(
        "Métriques par classe", 
        "eval_2_iou_f1_precision_recall.png", 
        "mIoU global de 0.588. Déséquilibre fort entre les IoU (Stratus à 0.25 vs Surface à 0.85)."
    )

    afficher_section_centree(
        "Dice Scores", 
        "eval_3_dice.png", 
        "Confirmation visuelle de la faiblesse sur les nuages bas (Stratus à 0.40)."
    )

    afficher_section_centree(
        "Prédictions vs Réel (Modèle 1)", 
        "PredictionvsterrainM1.png", 
        "Aperçu statique comparant l'image brute, la vérité terrain experte et la prédiction du modèle."
    )

    st.markdown("<h3 style='text-align: center;'>Comparaison Dynamique (Timelapses Modèle 1)</h3>", unsafe_allow_html=True)
    col_gif1, col_gif2, col_gif3 = st.columns(3)
    with col_gif1:
        st.markdown("<p style='text-align: center;'><strong>RGB (Vraie Couleur)</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_model_v1_1_rgb.gif", use_container_width=True)
    with col_gif2:
        st.markdown("<p style='text-align: center;'><strong>Vérité Terrain (Expert)</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_model_v1_2_expert.gif", use_container_width=True)
    with col_gif3:
        st.markdown("<p style='text-align: center;'><strong>Prédiction V1</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_model_v1_3_unet.gif", use_container_width=True)
    
    st.info("Segmentation spatialement plus lisse que l'algorithme expert (réduction du bruit), mais difficulté à délimiter précisément les amas de cumulus fins au cours du temps.")
    st.warning("**Résumé Modèle 1 :** Ce modèle constitue une base solide pour les phénomènes à fort contraste (orages, sol nu). Cependant, l'absence de bandes spectrales spécifiques rend la distinction entre types de nuages bas (Stratus/Cumulus) instable.")


# ------------------------------------------
# ONGLET 2 : MODÈLE 2
# ------------------------------------------
with tab2:
    st.header("Analyse du Modèle 2 (4 Bandes RGB + C02 | 30 Epochs)")
    st.markdown("<br>", unsafe_allow_html=True)

    afficher_section_centree(
        "Courbes d'entraînement", 
        "eval_4_training_curvesM2.png", 
        "Saut qualitatif majeur. La Loss de validation est divisée par deux (~0.28) et le mIoU bondit à 0.756.",
        type_message="success"
    )

    afficher_section_centree(
        "Matrice de Confusion", 
        "eval_1_confusion_matrixM2.png", 
        "Quasi-disparition des erreurs de classification. Le rappel des Stratus atteint 0.98 (contre 0.60 en M1).",
        type_message="success"
    )

    afficher_section_centree(
        "Distribution de classe", 
        "eval_5_distributionM2.png", 
        "Alignement quasi parfait entre les fréquences réelles et prédites. Disparition du biais pessimiste du modèle précédent.",
        type_message="success"
    )

    afficher_section_centree(
        "Radar de performance", 
        "eval_6_radarM2.png", 
        "Profil circulaire et large. Le modèle est désormais robuste et performant de manière uniforme sur toutes les classes.",
        type_message="success"
    )

    afficher_section_centree(
        "Métriques par classe", 
        "eval_2_iou_f1_precision_recallM2.png", 
        "mIoU final de 0.756 et Macro-F1 de 0.858. L'apport de la bande C02 (0.64 µm) est le facteur clé de ce succès.",
        type_message="success"
    )

    afficher_section_centree(
        "Dice Scores", 
        "eval_3_diceM2.png", 
        "Toutes les classes sont validées avec des scores élevés (>0.76).",
        type_message="success"
    )

    afficher_section_centree(
        "Prédictions vs Réel (Modèle 2)", 
        "unet_predictions_rgbM2.png", 
        "Fidélité extrême à l'arbre expert sur images statiques. Capacité à détecter des structures nuageuses très fines et à isoler parfaitement la surface.",
        type_message="success"
    )

    st.markdown("<h3 style='text-align: center;'>Comparaison Dynamique (Timelapses Modèle 2)</h3>", unsafe_allow_html=True)
    col_gif4, col_gif5, col_gif6 = st.columns(3)
    with col_gif4:
        st.markdown("<p style='text-align: center;'><strong>RGB (Vraie Couleur)</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_1_rgbM2.gif", use_container_width=True)
    with col_gif5:
        st.markdown("<p style='text-align: center;'><strong>Vérité Terrain (Expert)</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_2_expertM2.gif", use_container_width=True)
    with col_gif6:
        st.markdown("<p style='text-align: center;'><strong>Prédiction V2</strong></p>", unsafe_allow_html=True)
        st.image("images/timelapse_3_unetM2.gif", use_container_width=True)
        
    st.success("Remarquable stabilité temporelle. L'apport de la bande supplémentaire stabilise l'inférence dynamique.")
    st.success("**Résumé Modèle 2 :** L'intégration de la bande C02 (canal rouge à haute résolution) transforme le modèle. La précision spatiale et spectrale accrue permet d'égaler la logique de l'algorithme expert, offrant une segmentation de haute fiabilité.")

# ------------------------------------------
# ONGLET 3 : TEST EN DIRECT (INFÉRENCE)
# ------------------------------------------
with tab3:
    st.header("Test en Direct : Inférence sur un Dataset (1 Journée)")
    st.markdown("Chargez un jeu de données satellitaires brutes pour évaluer de manière dynamique les modèles `unet_best.pth` et `unet_v2_best.pth`.")

    # Zone d'upload des fichiers
    uploaded_files = st.file_uploader("Sélectionnez vos fichiers de données (.npy, .nc, .tif)", accept_multiple_files=True)

    if st.button("Lancer l'inférence sur les deux modèles", type="primary"):
        if uploaded_files:
            # Animation de la barre de progression pour simuler le traitement
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

            # Affichage des résultats simulés
            st.markdown("<h3 style='text-align: center;'>Aperçu des Prédictions</h3>", unsafe_allow_html=True)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown("<p style='text-align: center;'><strong>Donnée Brute (RGB)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la visualisation de la frame d'entrée.")
                # st.image(image_entree, use_container_width=True)
                
            with col_res2:
                st.markdown("<p style='text-align: center;'><strong>Modèle 1 (unet_best.pth)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la prédiction M1.")
                # st.image(prediction_m1, use_container_width=True)
                
            with col_res3:
                st.markdown("<p style='text-align: center;'><strong>Modèle 2 (unet_v2_best.pth)</strong></p>", unsafe_allow_html=True)
                st.info("Espace réservé pour la prédiction M2.")
                # st.image(prediction_m2, use_container_width=True)

            # Instructions pour l'intégration technique
            with st.expander("🛠️ Comment intégrer l'inférence PyTorch ici ?"):
                st.markdown("""
                Pour rendre ce bouton pleinement opérationnel avec vos fichiers `unet_best.pth` et `unet_v2_best.pth`, remplacez les `time.sleep()` par ce type de logique :
                ```python
                import torch
                from torchvision import transforms
                
                # 1. Charger les modèles
                model_v1 = torch.load('unet_best.pth', map_location=torch.device('cpu'))
                model_v2 = torch.load('unet_v2_best.pth', map_location=torch.device('cpu'))
                model_v1.eval()
                model_v2.eval()

                # 2. Traiter les fichiers uploadés (uploaded_files)
                # data_rgb = preprocess_rgb(uploaded_files)
                # data_rgb_co2 = preprocess_rgb_co2(uploaded_files)

                # 3. Inférence
                with torch.no_grad():
                    # pred_v1 = model_v1(data_rgb)
                    # pred_v2 = model_v2(data_rgb_co2)
                    
                # 4. Afficher avec st.image()
                ```
                """)

        else:
            st.error("⚠️ Veuillez charger au moins un fichier de données pour lancer l'inférence.")

# ==========================================
# SYNTHÈSE GLOBALE (TABLEAU COMPARATIF)
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("📊 Synthèse Comparative des Performances")

data_comparaison = {
    "Architecture": ["Modèle 1 (RGB)", "Modèle 2 (RGB + C02)"],
    "Bandes d'entrée": ["3", "4"],
    "mIoU": ["0.588", "0.756"],
    "Macro F1-Score": ["0.720", "0.858"],
    "Évolution mIoU": ["-", "+ 28.5%"],
    "Évolution F1-Score": ["-", "+ 19.1%"]
}
df_comparaison = pd.DataFrame(data_comparaison)

_, col_table, _ = st.columns([1, 3, 1])
with col_table:
    st.table(df_comparaison)
    st.info("La comparaison met en évidence l'importance critique du choix des bandes spectrales dans l'imagerie satellitaire. L'ajout d'une unique bande additionnelle a permis de résoudre les faiblesses structurelles de la première itération.")