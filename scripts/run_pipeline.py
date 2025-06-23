"""
Pipeline complet d'analyse automatisée du Vertebral Heart Score (VHS) en radiologie vétérinaire.

Ce module intègre un pipeline end-to-end comprenant:
1. Segmentation sémantique par réseau U-Net pour identification des structures anatomiques
2. Calcul automatique des métriques VHS par analyse géométrique (PCA)
3. Génération de rapports diagnostiques par IA (LLaMA 2)
4. Export multi-format (CSV + PDF individuels)

Architecture du pipeline:
- Preprocessing: Normalisation et redimensionnement des images à 512x512
- Segmentation: U-Net 14 classes pour structures thoraciques canines
- Analyse géométrique: Extraction des axes cardiaques par PCA et mesure vertébrale T4
- Diagnostic IA: Génération de rapports contextualisés via LLaMA 2
- Export: Sauvegarde structurée pour archivage et distribution

Classes de segmentation (supposées):
- Classe 1: Silhouette cardiaque
- Classe 8: Vertèbre T4 (référence anatomique)
- Classes 0-13: Autres structures thoraciques

Dépendances:
    - torch: Framework de deep learning
    - torchvision: Transformations d'images
    - PIL: Traitement d'images Python
    - opencv-python: Analyse géométrique et contours
    - pandas: Manipulation de données
    - reportlab: Génération PDF
    - ollama: Interface LLaMA 2
    - tqdm: Barres de progression
    - numpy: Calculs numériques

Auteur: [À compléter]
Version: 2.0 - Pipeline intégré
Date: [À compléter]
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import ollama
from src.models.unet import UNet

# === Configuration du pipeline ===
# Répertoire source contenant les images radiographiques à analyser
IMAGE_DIR = "data/valid"

# Fichier CSV de consolidation des résultats quantitatifs
OUTPUT_CSV = "vhs_results_batch.csv"

# Répertoire de destination pour les rapports PDF individuels
REPORTS_DIR = "generated_reports/pdf"

# Chemin vers le modèle U-Net pré-entraîné pour la segmentation thoracique
MODEL_PATH = "unet_model.pth"

# === Initialisation de l'environnement ===
# Création du répertoire de sortie avec gestion des permissions
os.makedirs(REPORTS_DIR, exist_ok=True)

# === Chargement du modèle de segmentation ===
print("🔄 Loading model...")

# Initialisation de l'architecture U-Net
# - 3 canaux d'entrée (RGB)
# - 14 classes de sortie (structures anatomiques thoraciques)
model = UNet(in_channels=3, out_channels=14)

# Chargement des poids pré-entraînés avec mappage CPU pour compatibilité
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# Passage en mode évaluation (désactivation dropout/batch norm)
model.eval()
print("✅ Model loaded.")

# === Pipeline de préprocessing des images ===
# Transformation standardisée pour l'inférence du modèle
transform = T.Compose([
    T.Resize((512, 512)),    # Redimensionnement à la résolution d'entraînement
    T.ToTensor()             # Conversion PIL -> Tensor avec normalisation [0,1]
])

# === Fonction de calcul VHS par analyse géométrique ===
def compute_vhs_from_mask(mask):
    """
    Calcule le Vertebral Heart Score à partir d'un masque de segmentation.
    
    Le VHS est un indicateur clinique majeur de cardiomégalie chez les chiens,
    calculé comme le ratio (axe_long + axe_court) / hauteur_T4.
    
    Méthodologie:
    1. Extraction des pixels cardiaques (classe 1) et vertébraux T4 (classe 8)
    2. Analyse en Composantes Principales (PCA) pour orientation cardiaque optimale
    3. Calcul des axes principaux par projection sur les vecteurs propres
    4. Mesure de la hauteur T4 par bounding box du contour principal
    
    Args:
        mask (numpy.ndarray): Masque de segmentation 2D avec classes étiquetées
        
    Returns:
        tuple: (vhs_score, long_axis, short_axis, t4_height) ou (None, None, None, None)
        
    Notes:
        - Retourne None si structures manquantes (échec de segmentation)
        - La PCA assure une mesure robuste indépendante de l'orientation image
    """
    import cv2
    
    # Extraction des masques binaires pour structures d'intérêt
    heart_mask = (mask == 1).astype(np.uint8)    # Silhouette cardiaque
    t4_mask = (mask == 8).astype(np.uint8)       # Vertèbre T4 de référence

    # Validation de la présence des structures anatomiques critiques
    if heart_mask.sum() == 0 or t4_mask.sum() == 0:
        return None, None, None, None

    # === Analyse géométrique cardiaque par PCA ===
    # Extraction des coordonnées des pixels cardiaques
    coords = np.column_stack(np.where(heart_mask > 0))
    
    # Calcul de l'Analyse en Composantes Principales
    # Permet d'identifier les axes principaux de la silhouette cardiaque
    mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
    center = mean[0]
    
    # Projection des coordonnées sur les axes principaux
    projections = (coords - center) @ eigenvectors.T
    
    # Calcul des dimensions cardiaques selon les axes principaux
    long_axis = np.ptp(projections[:, 0])   # Axe principal (plus grande variance)
    short_axis = np.ptp(projections[:, 1])  # Axe secondaire (variance orthogonale)

    # === Mesure de référence vertébrale T4 ===
    # Détection des contours de la vertèbre T4
    contours, _ = cv2.findContours(t4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sélection du contour principal (plus grande surface)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Extraction de la hauteur via bounding box
    # Index 3 correspond à la hauteur dans (x, y, width, height)
    t4_h = cv2.boundingRect(main_contour)[3]

    # Validation de la mesure de référence
    if t4_h == 0:
        return None, None, None, None

    # === Calcul du score VHS normalisé ===
    # Formule clinique standard: (LA + SA) / T4_height
    vhs = (long_axis + short_axis) / t4_h
    
    return vhs, long_axis, short_axis, t4_h

# === Générateur de rapports diagnostiques par IA ===
def generate_report(vhs, long_axis, short_axis, t4_height):
    """
    Génère un rapport diagnostique structuré via LLaMA 2.
    
    Utilise l'intelligence artificielle pour contextualiser les mesures
    quantitatives en interprétation clinique vétérinaire.
    
    Args:
        vhs (float): Score VHS calculé
        long_axis (float): Axe cardiaque principal en pixels
        short_axis (float): Axe cardiaque secondaire en pixels
        t4_height (float): Hauteur de référence T4 en pixels
        
    Returns:
        str: Rapport diagnostique formaté en langage naturel
        
    Notes:
        - Prompt optimisé pour expertise radiologique vétérinaire
        - Précision numérique adaptée à l'usage clinique
    """
    # Construction du prompt médical contextualisé
    prompt = f"""
You are a veterinary radiologist. Based on the following measurements, write a diagnostic report:
- VHS: {vhs:.2f}
- Long axis: {long_axis:.1f}
- Short axis: {short_axis:.1f}
- T4 height: {t4_height:.1f}
"""
    
    # Appel au modèle LLaMA 2 via l'API Ollama
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    
    # Extraction du contenu textuel de la réponse
    return response["message"]["content"]

# === Fonction d'export PDF individuel ===
def save_pdf(filename, content_lines):
    """
    Sauvegarde un rapport sous format PDF avec mise en page professionnelle.
    
    Reprend la logique du module de génération PDF avec:
    - Format A4 standard
    - Police Helvetica 11pt
    - Gestion automatique des sauts de page
    - Marges professionnelles
    
    Args:
        filename (str): Nom du fichier PDF (sans extension)
        content_lines (list): Lignes de contenu à intégrer
        
    Side Effects:
        Crée un fichier PDF dans REPORTS_DIR
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    
    # Construction du chemin de destination
    pdf_path = os.path.join(REPORTS_DIR, f"{filename}.pdf")
    
    # Initialisation du canvas ReportLab
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    # Configuration initiale de la mise en page
    y = height - 50  # Marge supérieure
    c.setFont("Helvetica", 11)
    
    # Traitement séquentiel des lignes avec gestion des pages
    for line in content_lines:
        # Vérification de l'espace disponible
        if y < 40:  # Marge inférieure de sécurité
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)  # Réapplication après saut de page
        
        # Insertion de la ligne avec nettoyage des espaces
        c.drawString(40, y, line.strip())
        y -= 18  # Interligne standard
    
    # Finalisation du document
    c.save()

# === Pipeline principal de traitement par lots ===
# Liste d'accumulation des résultats pour export CSV
results = []

# Découverte automatique des images dans le répertoire source
image_list = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
print(f"🔍 Found {len(image_list)} image(s) to process.")

# === Boucle de traitement avec suivi de progression ===
for fname in tqdm(image_list, desc="🧠 Processing Images"):
    try:
        # === Étape 1: Chargement et préprocessing ===
        img_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(img_path).convert("RGB")  # Normalisation colorimétrique
        input_tensor = transform(image).unsqueeze(0)  # Ajout dimension batch
        
        # === Étape 2: Inférence de segmentation ===
        with torch.no_grad():  # Désactivation du calcul de gradients pour l'inférence
            output = model(input_tensor)
            # Conversion logits -> classes par argmax sur la dimension des canaux
            pred = torch.argmax(output, dim=1).squeeze().numpy()
        
        # === Étape 3: Analyse géométrique et calcul VHS ===
        vhs, la, sa, t4 = compute_vhs_from_mask(pred)
        
        # === Étape 4: Traitement conditionnel des résultats valides ===
        if vhs:  # Vérification de la validité des mesures
            print(f"✅ Processed {fname} | VHS: {vhs:.2f}")
            
            # === Étape 5: Génération du rapport diagnostique ===
            report_text = generate_report(vhs, la, sa, t4)
            
            # === Étape 6: Export PDF individuel ===
            # Nettoyage du nom de fichier et conversion en lignes pour PDF
            clean_filename = fname.replace(".jpg", "").replace(".png", "")
            save_pdf(clean_filename, report_text.splitlines())
            
            # === Étape 7: Accumulation des données structurées ===
            results.append({
                "image_id": fname,
                "vhs": round(vhs, 2),                    # Précision clinique appropriée
                "heart_long_axis": round(la, 2),
                "heart_short_axis": round(sa, 2),
                "t4_height": round(t4, 2),
                "report": report_text                    # Rapport complet pour archivage
            })
        else:
            # Gestion des échecs de segmentation
            print(f"[!] Skipped {fname} (missing heart or T4)")
    
    except Exception as e:
        # Gestion robuste des erreurs avec continuation du traitement
        print(f"[X] Error processing {fname}: {e}")

# === Export consolidé des résultats ===
# Sauvegarde du dataset complet au format CSV pour analyse ultérieure
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(results)} reports to {OUTPUT_CSV}")