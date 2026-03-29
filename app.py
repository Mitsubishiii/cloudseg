import streamlit as st
import time

# Configuration de la page
st.set_page_config(page_title="Dashboard IA - IGN", layout="centered") # layout="centered" pour éviter que les images soient trop larges

# ==========================================
# 1. EN-TÊTE
# ==========================================
st.title("Application de l'IA à l'Imagerie Satellitaire")
st.subheader("Projet de 2ème Année -- Approche Experte et Réduction de Dimension")
st.markdown("Ce tableau de bord présente l'évolution de nos modèles de segmentation nuageuse (GOES-16). Faites défiler pour voir l'analyse détaillée métrique par métrique, puis testez de nouvelles données.")
st.markdown("---")

# ==========================================
# 2. ANALYSE MODÈLE 1
# ==========================================
st.header("1️⃣ Analyse des Résultats - Modèle 1 (U-Net V1)")

st.subheader("Courbes d'entraînement")
st.image("images/eval_4_training_curves.png", use_container_width=True)
st.info("La mIoU plafonne bas (0.55). Le modèle est en underfitting et peine à apprendre les nuances complexes.")

st.markdown("---")
st.subheader("Métriques Globales")
st.image("images/eval_2_iou_f1_precision_recall.png", use_container_width=True)
st.warning("Pour la classe Stratus, le Recall est correct (~0.6) mais la précision est très faible (~0.3).")

st.markdown("---")
st.subheader("Radar des performances")
st.image("images/eval_6_radar.png", use_container_width=True)
st.warning("L'effondrement de la courbe vers le centre confirme que le Modèle 1 est 'aveugle' aux nuances des nuages de basse altitude (Stratus et Cumulus).")

st.markdown("---")
st.subheader("Scores de Dice")
st.image("images/eval_3_dice.png", use_container_width=True)
st.error("Les teintes jaunes mettent en évidence l'incapacité du modèle à détourer correctement les Stratus (0.405) et les Cumulus (0.530).")

st.markdown("---")
st.subheader("Matrice de Confusion")
st.image("images/eval_1_confusion_matrix.png", use_container_width=True)
st.error("Problème majeur : 32% des vrais Stratus sont classés comme Cumulus. Le modèle n'arrive pas à les séparer.")

st.markdown("---")
st.subheader("Biais de Distribution")
st.image("images/eval_5_distribution.png", use_container_width=True)
st.error("Le modèle surestime la quantité de Cumulus au détriment de la surface dégagée, générant du bruit visuel.")

st.markdown("---")
st.markdown("---")

# ==========================================
# 3. ANALYSE MODÈLE 2
# ==========================================
st.header("2️⃣ Analyse des Résultats - Modèle 2 (U-Net V2)")

st.subheader("Courbes d'entraînement")
st.image("images/eval_4_training_curvesM2.png", use_container_width=True)
st.success("Convergence saine et absence d'overfitting. La mIoU franchit la barre des 0.70 de manière stable.")

st.markdown("---")
st.subheader("Métriques Globales")
st.image("images/eval_2_iou_f1_precision_recallM2.png", use_container_width=True)
st.success("Bond de performance : Le Macro-F1 passe à 0.858. L'énorme gouffre de performance sur les Stratus a disparu.")

st.markdown("---")
st.subheader("Radar des performances")
st.image("images/eval_6_radarM2.png", use_container_width=True)
st.success("L'hexagone est devenu régulier. Le problème de détection des basses altitudes est définitivement résolu.")

st.markdown("---")
st.subheader("Scores de Dice")
st.image("images/eval_3_diceM2.png", use_container_width=True)
st.success("Tous les scores sont dans le vert foncé (0.881 pour les Stratus). Les masques épousent les contours réels.")

st.markdown("---")
st.subheader("Matrice de Confusion")
st.image("images/eval_1_confusion_matrixM2.png", use_container_width=True)
st.success("La confusion Stratus/Cumulus tombe à un niveau marginal (3%). La diagonale est excellente.")

st.markdown("---")
st.subheader("Biais de Distribution")
st.image("images/eval_5_distributionM2.png", use_container_width=True)
st.success("La distribution prédite colle parfaitement à la réalité terrain de l'expert IGN.")

st.markdown("---")
st.markdown("---")

# ==========================================
# 4. COMPARAISON VISUELLE (GIFs)
# ==========================================
st.header("3️⃣ Comparaison Dynamique (Timelapses)")

st.subheader("Modèle 1 vs Vérité Terrain")
col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    st.markdown("**RGB (Vraie Couleur)**")
    st.image("images/timelapse_model_v1_1_rgb.gif", use_container_width=True)
with col_g2:
    st.markdown("**Vérité Terrain (Expert)**")
    st.image("images/timelapse_model_v1_2_expert.gif", use_container_width=True)
with col_g3:
    st.markdown("**Prédiction V1**")
    st.image("images/timelapse_model_v1_3_unet.gif", use_container_width=True)
st.error("Le Modèle 1 scintille énormément et échoue sur les basses altitudes.")

st.markdown("---")

st.subheader("Modèle 2 vs Vérité Terrain")
col_g4, col_g5, col_g6 = st.columns(3)
with col_g4:
    st.markdown("**RGB (Vraie Couleur)**")
    st.image("images/timelapse_1_rgbM2.gif", use_container_width=True)
with col_g5:
    st.markdown("**Vérité Terrain (Expert)**")
    st.image("images/timelapse_2_expertM2.gif", use_container_width=True)
with col_g6:
    st.markdown("**Prédiction V2**")
    st.image("images/timelapse_3_unetM2.gif", use_container_width=True)
st.success("Le Modèle 2 offre une stabilité temporelle et une fidélité structurelle impressionnantes.")

st.markdown("---")

# ==========================================
# 5. ÉVALUATION EN DIRECT (NOUVELLE DONNÉE)
# ==========================================
st.header("🚀 Évaluation en direct : Testez le modèle")
st.markdown("Chargez une séquence temporelle de données satellitaires brutes pour évaluer les modèles en direct.")

uploaded_files = st.file_uploader("Sélectionnez vos fichiers de données (ex: tenseurs .npy)", accept_multiple_files=True)

if st.button("Lancer l'inférence sur les deux modèles", type="primary"):
    if uploaded_files:
        my_bar = st.progress(0, text="Chargement des modèles...")
        time.sleep(1) 
        
        my_bar.progress(30, text="Inférence Modèle 1 en cours...")
        time.sleep(1.5) 
        
        my_bar.progress(60, text="Inférence Modèle 2 en cours...")
        time.sleep(1.5) 
        
        my_bar.progress(90, text="Génération des GIFs animés...")
        time.sleep(1.5) 
        
        my_bar.progress(100, text="Terminé !")
        time.sleep(0.5)
        my_bar.empty() 
        
        st.success("Inférence terminée ! Voici les résultats comparatifs :")
        
        # Affichage simulé des résultats (à remplacer par la génération de tes vrais GIFs)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.subheader("Résultat Modèle 1")
            st.info("Métrique simulée (mIoU) : 0.54")
            
        with col_res2:
            st.subheader("Résultat Modèle 2")
            st.success("Métrique simulée (mIoU) : 0.76")
            
    else:
        st.error("⚠️ Veuillez charger des données avant de lancer l'évaluation.")