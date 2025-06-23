"""
Application Streamlit pour visualiser et analyser les résultats VHS (Vertebral Heart Scale).

Cette application web permet de :
1. Visualiser les résultats VHS calculés à partir d'un fichier CSV
2. Filtrer les résultats par sévérité (Normal/Mild/High)
3. Afficher les images radiographiques avec leurs rapports
4. Télécharger les rapports en PDF
5. Uploader de nouvelles radiographies pour analyse en temps réel

Interface utilisateur :
- Sidebar : filtres par sévérité
- Section principale : tableau des résultats, graphiques, visualisation d'images
- Section upload : analyse de nouvelles images avec génération de rapport IA

Dépendances :
- streamlit : interface web
- pandas : manipulation des données
- matplotlib : visualisations
- torch : inférence du modèle U-Net
- ollama : génération de rapports IA
- PIL, cv2 : traitement d'images
"""

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

# Ajout du chemin parent pour importer les modules locaux
# Permet d'accéder aux modules dans src/ depuis l'interface web
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.unet import UNet

# ============= CONFIGURATION DES CHEMINS =============
# Chemins vers les fichiers de données et résultats
CSV_PATH = "vhs_results.csv"              # Fichier CSV avec les résultats VHS
REPORTS_DIR = "generated_reports/pdf"     # Dossier contenant les rapports PDF
TXT_REPORT = "vhs_llama2_reports.txt"     # Fichier texte avec tous les rapports
IMAGE_DIR = "data/valid"                  # Dossier contenant les images de validation

print("📁 Configuration des chemins :")
print(f"  • CSV des résultats : {CSV_PATH}")
print(f"  • Rapports PDF : {REPORTS_DIR}")
print(f"  • Rapports texte : {TXT_REPORT}")
print(f"  • Images : {IMAGE_DIR}")

# ============= CHARGEMENT DES DONNÉES CSV =============
# Lecture du fichier CSV contenant les résultats VHS calculés
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV chargé : {len(df)} résultats VHS trouvés")
except FileNotFoundError:
    st.error(f"❌ Fichier CSV non trouvé : {CSV_PATH}")
    st.stop()

# ============= FONCTION DE CLASSIFICATION VHS =============
def vhs_color(v):
    """
    Classe les valeurs VHS selon leur sévérité clinique.
    
    Le VHS (Vertebral Heart Scale) est interprété selon les seuils vétérinaires :
    - Normal : VHS < 9.0 (cœur de taille normale)
    - Mild : 9.0 ≤ VHS ≤ 10.5 (légère cardiomégalie)
    - High : VHS > 10.5 (cardiomégalie significative)
    
    Args:
        v (float): Valeur VHS calculée
    
    Returns:
        str: Classification avec emoji pour l'interface utilisateur
    """
    if v < 9.0:
        return "🟢 Normal"      # Cœur de taille normale
    elif v <= 10.5:
        return "🟡 Mild"        # Cardiomégalie légère
    else:
        return "🔴 High"        # Cardiomégalie sévère

# Application de la classification à toutes les valeurs VHS
df["Severity"] = df["vhs"].apply(vhs_color)
print(f"📊 Classification appliquée : {df['Severity'].value_counts().to_dict()}")

# ============= FONCTION DE CHARGEMENT DES RAPPORTS =============
def load_reports(txt_path):
    """
    Parse un fichier texte contenant plusieurs rapports VHS.
    
    Le fichier est structuré avec :
    - "Image: nom_fichier.jpg" comme séparateur
    - Contenu du rapport sur les lignes suivantes
    - Répétition pour chaque image
    
    Args:
        txt_path (str): Chemin vers le fichier texte des rapports
    
    Returns:
        dict: Dictionnaire {nom_image_sans_extension: contenu_rapport}
    
    Example:
        Fichier d'entrée :
        ```
        Image: dog001.jpg
        VHS: 8.5 - Normal cardiac silhouette...
        
        Image: dog002.jpg  
        VHS: 11.2 - Moderate cardiomegaly...
        ```
        
        Sortie :
        ```
        {
            "dog001": "Image: dog001.jpg\nVHS: 8.5 - Normal...",
            "dog002": "Image: dog002.jpg\nVHS: 11.2 - Moderate..."
        }
        ```
    """
    reports = {}
    
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"⚠️  Fichier de rapports non trouvé : {txt_path}")
        return {}

    current = ""           # Nom de l'image courante
    buffer = []           # Buffer pour accumuler les lignes du rapport

    for line in lines:
        if line.startswith("Image:"):
            # Nouvelle image détectée : sauvegarder le rapport précédent
            if current:
                reports[current] = "".join(buffer).strip()
            
            # Extraction du nom de fichier sans extension
            current = line.split("Image:")[1].strip().replace(".jpg", "").replace(".png", "")
            buffer = [line]  # Initialiser le nouveau buffer
        else:
            # Ligne de contenu : ajouter au buffer
            buffer.append(line)
    
    # Sauvegarder le dernier rapport
    if current and buffer:
        reports[current] = "".join(buffer).strip()
    
    print(f"📄 Rapports chargés : {len(reports)} rapports trouvés")
    return reports

# Chargement des rapports texte
report_dict = load_reports(TXT_REPORT)

# ============= INTERFACE STREAMLIT PRINCIPALE =============
# Configuration de la page
st.set_page_config(
    page_title="VHS Report Viewer",
    page_icon="🦥",
    layout="wide"
)

# Titre principal avec émojis pour l'interface utilisateur
st.title("🦥 VHS Report Viewer")
st.markdown("View segmentation results and generated reports.")
st.markdown("---")

# ============= SECTION UPLOAD DE NOUVELLES IMAGES =============
st.subheader("📥 Upload a New X-ray")
st.markdown("Upload a chest X-ray image to compute VHS in real-time")

# Widget d'upload de fichier avec types supportés
uploaded_file = st.file_uploader("Upload an image (.jpg or .png)", type=["jpg", "png"])

if uploaded_file:
    # === Imports spécifiques pour l'analyse en temps réel ===
    from PIL import Image
    import torch
    import numpy as np
    import torchvision.transforms as T
    import ollama

    @st.cache_resource
    def load_model():
        """
        Charge le modèle U-Net pré-entraîné pour la segmentation.
        
        Utilise le cache de Streamlit (@st.cache_resource) pour éviter
        de recharger le modèle à chaque interaction utilisateur.
        
        Returns:
            UNet: Modèle PyTorch en mode évaluation
        """
        from src.models.unet import UNet
        
        # Initialisation du modèle avec 14 classes de sortie
        model = UNet(in_channels=3, out_channels=14)
        
        # Chargement des poids pré-entraînés sur CPU pour compatibilité
        model.load_state_dict(torch.load("unet_model.pth", map_location="cpu"))
        
        # Mode évaluation pour l'inférence
        model.eval()
        return model

    # === Chargement et préparation de l'image ===
    model = load_model()
    
    # Conversion en RGB pour assurer la compatibilité
    image = Image.open(uploaded_file).convert("RGB")
    
    # Affichage de l'image uploadée
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # === Préprocessing de l'image ===
    # Pipeline de transformation pour l'inférence
    transform = T.Compose([
        T.Resize((512, 512)),  # Redimensionnement à la taille d'entraînement
        T.ToTensor()           # Conversion en tensor [0,1] + normalisation
    ])
    
    # Application des transformations + ajout de la dimension batch
    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 512, 512)

    # === Inférence du modèle ===
    with torch.no_grad():  # Pas de calcul de gradients pour économiser la mémoire
        output = model(input_tensor)                    # Logits: (1, 14, 512, 512)
        pred = torch.argmax(output, dim=1).squeeze().numpy()  # Classes: (512, 512)

    # === Fonction de calcul VHS intégrée ===
    def compute_vhs_from_mask(mask):
        """
        Calcule le VHS à partir du masque de segmentation prédit.
        
        Version optimisée de la fonction originale pour l'interface web.
        Utilise np.ptp() pour calculer les étendues plus efficacement.
        
        Args:
            mask (np.ndarray): Masque de segmentation avec classes [0-13]
        
        Returns:
            tuple: (vhs, long_axis, short_axis, t4_height) ou (None, None, None, None)
        """
        import cv2
        
        # Extraction des masques binaires
        heart_mask = (mask == 1).astype(np.uint8)   # Classe 1 = cœur
        t4_mask = (mask == 8).astype(np.uint8)      # Classe 8 = vertèbre T4

        # Vérification de la présence des structures
        if heart_mask.sum() == 0 or t4_mask.sum() == 0:
            return None, None, None, None

        # === Calcul des axes du cœur via PCA ===
        coords = np.column_stack(np.where(heart_mask > 0))
        mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
        center = mean[0]
        
        projections = (coords - center) @ eigenvectors.T
        # Utilisation de np.ptp (peak-to-peak) pour l'étendue
        long_axis = np.ptp(projections[:, 0])    # max - min
        short_axis = np.ptp(projections[:, 1])

        # === Calcul de la hauteur T4 ===
        contours, _ = cv2.findContours(t4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t4_h = cv2.boundingRect(max(contours, key=cv2.contourArea))[3]  # Hauteur de la bbox

        if t4_h == 0:
            return None, None, None, None

        # === Calcul final du VHS ===
        vhs = (long_axis + short_axis) / t4_h
        return vhs, long_axis, short_axis, t4_h

    # === Calcul du VHS sur l'image uploadée ===
    vhs, long_axis, short_axis, t4_height = compute_vhs_from_mask(pred)

    if vhs:
        # === Affichage du résultat VHS ===
        st.success(f"✅ VHS: {vhs:.2f}")
        
        # Affichage des composantes pour le diagnostic
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Long Axis", f"{long_axis:.1f} px")
        with col2:
            st.metric("Short Axis", f"{short_axis:.1f} px")
        with col3:
            st.metric("T4 Height", f"{t4_height:.1f} px")

        # === Génération du rapport IA avec Ollama ===
        # Construction du prompt pour le modèle de langage
        prompt = f"""
You are a veterinary radiologist. Based on the following measurements, write a diagnostic report:

- VHS: {vhs:.2f}
- Long axis: {long_axis:.1f}
- Short axis: {short_axis:.1f}
- T4 height: {t4_height:.1f}
"""

        # Appel à l'API Ollama pour générer le rapport
        try:
            with st.spinner("Generating AI report..."):
                response = ollama.chat(
                    model="llama2",
                    messages=[{"role": "user", "content": prompt}]
                )

            # Affichage du rapport généré
            st.subheader("📄 Generated Report")
            st.text(response['message']['content'])
            
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            
    else:
        # Cas d'échec : structures non détectées
        st.error("❌ Could not compute VHS — heart or T4 not detected.")
        st.info("💡 Make sure the X-ray shows both the heart and T4 vertebra clearly.")

# ============= SECTION FILTRAGE DES DONNÉES =============
st.markdown("---")
st.subheader("📊 VHS Results Analysis")

# Sidebar pour les filtres
st.sidebar.header("🔍 Filters")
selected_severity = st.sidebar.selectbox(
    "Filter by severity", 
    ["All", "🟢 Normal", "🟡 Mild", "🔴 High"],
    help="Filter results by VHS severity classification"
)

# Application du filtre si sélectionné
df_filtered = df.copy()
if selected_severity != "All":
    df_filtered = df[df["Severity"] == selected_severity]
    st.info(f"Showing {len(df_filtered)} results for severity: {selected_severity}")

# ============= AFFICHAGE DU TABLEAU DE DONNÉES =============
st.subheader("📋 VHS Results Table")

# Tableau interactif avec colonnes sélectionnées
st.dataframe(
    df_filtered[["image_id", "vhs", "heart_long_axis", "heart_short_axis", "t4_height", "Severity"]].reset_index(drop=True),
    use_container_width=True
)

# Statistiques rapides
if len(df_filtered) > 0:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(df_filtered))
    with col2:
        st.metric("Average VHS", f"{df_filtered['vhs'].mean():.2f}")
    with col3:
        st.metric("Min VHS", f"{df_filtered['vhs'].min():.2f}")
    with col4:
        st.metric("Max VHS", f"{df_filtered['vhs'].max():.2f}")

# ============= GRAPHIQUE DE DISTRIBUTION VHS =============
st.subheader("📊 VHS Score Distribution")

# Création du graphique avec matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Histogramme avec personnalisation
ax.hist(df_filtered["vhs"], bins=10, color="skyblue", edgecolor="black", alpha=0.7)

# Ajout des lignes de seuil cliniques
ax.axvline(x=9.0, color='green', linestyle='--', alpha=0.7, label='Normal threshold')
ax.axvline(x=10.5, color='orange', linestyle='--', alpha=0.7, label='High threshold')

# Configuration des axes et titre
ax.set_xlabel("VHS Score")
ax.set_ylabel("Number of Images")
ax.set_title("Distribution of VHS Scores")
ax.legend()
ax.grid(True, alpha=0.3)

# Affichage dans Streamlit
st.pyplot(fig)

# ============= SECTION VISUALISATION D'IMAGES ET RAPPORTS =============
st.markdown("---")
st.subheader("🖼️ Image Viewer & Reports")

# Sélection d'image depuis le tableau filtré
if len(df_filtered) > 0:
    selected_image = st.selectbox(
        "Select an image to view its report", 
        df_filtered["image_id"],
        help="Choose an image to display it with its corresponding VHS report"
    )

    # === Affichage de l'image sélectionnée ===
    image_path = os.path.join(IMAGE_DIR, selected_image)
    if os.path.exists(image_path):
        # Image trouvée : affichage avec informations VHS
        st.image(image_path, caption=f"X-ray: {selected_image}", use_column_width=True)
        
        # Affichage des métriques VHS pour l'image sélectionnée
        selected_row = df_filtered[df_filtered["image_id"] == selected_image].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VHS", f"{selected_row['vhs']:.2f}")
        with col2:
            st.metric("Long Axis", f"{selected_row['heart_long_axis']:.1f}")
        with col3:
            st.metric("Short Axis", f"{selected_row['heart_short_axis']:.1f}")
        with col4:
            st.metric("T4 Height", f"{selected_row['t4_height']:.1f}")
            
    else:
        st.warning(f"⚠️  Image not found: {image_path}")

    # === Affichage du rapport textuel ===
    # Conversion du nom de fichier en clé de rapport (sans extension)
    report_key = selected_image.replace(".jpg", "").replace(".png", "")
    
    if report_key in report_dict:
        st.subheader(f"📄 Report for {selected_image}")
        
        # Affichage du rapport dans une zone de texte stylisée
        st.text_area(
            "Detailed VHS Report",
            report_dict[report_key],
            height=200,
            disabled=True  # Lecture seule
        )
    else:
        st.warning("⚠️  No report found for this image.")

    # === Téléchargement du rapport PDF ===
    pdf_file = os.path.join(REPORTS_DIR, f"{report_key}.pdf")
    if os.path.exists(pdf_file):
        # Bouton de téléchargement si le PDF existe
        with open(pdf_file, "rb") as f:
            st.download_button(
                "📥 Download PDF Report",
                f,
                file_name=os.path.basename(pdf_file),
                mime="application/pdf"
            )
    else:
        st.info("💡 PDF report not available for this image.")

else:
    st.warning("⚠️  No data available with the current filter.")

# ============= FOOTER INFORMATIF =============
st.markdown("---")
st.markdown("""
### 📖 About VHS (Vertebral Heart Scale)

The Vertebral Heart Scale is a radiographic measurement used in veterinary medicine to assess cardiac size:
- **Normal**: VHS < 9.0 (normal heart size)
- **Mild cardiomegaly**: 9.0 ≤ VHS ≤ 10.5 (slight heart enlargement)  
- **Significant cardiomegaly**: VHS > 10.5 (significant heart enlargement)

**Calculation**: VHS = (Long axis + Short axis) / T4 vertebra height
""")