"""
Script de calcul du VHS (Vertebral Heart Scale) à partir de masques de segmentation.

Le VHS est un indice radiologique utilisé en médecine vétérinaire pour évaluer 
la taille du cœur par rapport à la colonne vertébrale. Il se calcule comme :
VHS = (axe_long_coeur + axe_court_coeur) / hauteur_vertebre_T4

Ce script :
1. Charge un modèle U-Net pré-entraîné
2. Effectue la segmentation sur un dataset de validation
3. Calcule le VHS pour chaque image
4. Sauvegarde les résultats dans un fichier CSV

Classes de segmentation :
- 0 : arrière-plan
- 1 : cœur
- 2-13 : vertèbres (T4 = classe 8)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import VALID_DIR, VALID_ANNOTATIONS
from src.utils.data_loader import get_dataloader
from src.models.unet import UNet
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
import cv2
import os
import csv

# ============= PARAMÈTRES DE CONFIGURATION =============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # Batch size = 1 pour un calcul précis pixel par pixel du VHS
NUM_CLASSES = 14  # Classes : 0 = arrière-plan, 1 = cœur, 2-13 = vertèbres

print(f"Device utilisé : {DEVICE}")
print(f"Nombre de classes : {NUM_CLASSES}")

# ============= CHARGEMENT DU MODÈLE =============
# Initialisation du modèle U-Net avec les paramètres d'entraînement
model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)

# Chargement des poids pré-entraînés
model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))

# Mode évaluation : désactive dropout et batch norm pour l'inférence
model.eval()
print("✅ Modèle chargé et configuré en mode évaluation")

# ============= CHARGEMENT DES DONNÉES =============
# Création du dataloader pour le dataset de validation
# shuffle=False pour un traitement ordonné et reproductible
val_loader = get_dataloader(VALID_DIR, VALID_ANNOTATIONS, batch_size=BATCH_SIZE, shuffle=False)
print(f"✅ Dataset de validation chargé : {len(val_loader)} images")

# ============= FONCTION DE CALCUL VHS =============
def compute_vhs_from_mask(mask):
    """
    Calcule le VHS (Vertebral Heart Scale) à partir d'un masque de segmentation.
    
    Le VHS est calculé selon la formule :
    VHS = (axe_long_cœur + axe_court_cœur) / hauteur_vertèbre_T4
    
    Étapes :
    1. Extraction des masques du cœur (classe 1) et de T4 (classe 8)
    2. Calcul des axes du cœur via l'Analyse en Composantes Principales (PCA)
    3. Mesure de la hauteur de la vertèbre T4 via bounding box
    4. Calcul du ratio VHS
    
    Args:
        mask (np.ndarray): Masque de segmentation avec classes [0-13]
    
    Returns:
        float or None: Valeur VHS ou None si calcul impossible
    """
    # Extraction des masques binaires pour cœur et T4
    heart_mask = (mask == 1).astype(np.uint8)  # Classe 1 = cœur
    t4_mask = (mask == 8).astype(np.uint8)     # Classe 8 = vertèbre T4

    # Vérification de la présence des structures anatomiques nécessaires
    if heart_mask.sum() == 0 or t4_mask.sum() == 0:
        return None

    # === ÉTAPE 1: Calcul des axes du cœur via PCA ===
    # Récupération des coordonnées de tous les pixels du cœur
    coords = np.column_stack(np.where(heart_mask > 0))
    
    # Application de l'Analyse en Composantes Principales
    # PCA permet de trouver les axes principaux de la forme du cœur
    mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
    center = mean[0]  # Centre de masse du cœur

    # Projection des coordonnées sur les axes principaux
    projections = (coords - center) @ eigenvectors.T
    
    # Calcul des longueurs des axes (différence max-min sur chaque composante)
    heart_long_axis = projections[:, 0].max() - projections[:, 0].min()   # 1ère composante = axe long
    heart_short_axis = projections[:, 1].max() - projections[:, 1].min()  # 2ème composante = axe court

    # === ÉTAPE 2: Calcul de la hauteur de la vertèbre T4 ===
    # Détection des contours de la vertèbre T4
    contours, _ = cv2.findContours(t4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sélection du plus grand contour (au cas où il y aurait du bruit)
    t4_contour = max(contours, key=cv2.contourArea)
    
    # Calcul de la bounding box pour obtenir les dimensions
    _, _, w, h = cv2.boundingRect(t4_contour)
    t4_height = h  # Hauteur de la vertèbre T4

    # Vérification de la validité de la mesure
    if t4_height == 0:
        return None

    # === ÉTAPE 3: Calcul final du VHS ===
    # Formule standard : VHS = (axe_long + axe_court) / hauteur_T4
    vhs = (heart_long_axis + heart_short_axis) / t4_height
    return vhs


# ============= CONFIGURATION DE LA SAUVEGARDE =============
# Nom du fichier de sortie pour les résultats
output_file = "vhs_results.csv"

# Définition des colonnes du fichier CSV
fieldnames = ["image_id", "vhs", "heart_long_axis", "heart_short_axis", "t4_height"]

# Liste pour stocker tous les résultats
results = []

print(f"\n🔄 Début du calcul VHS sur {len(val_loader)} images...")

# ============= BOUCLE PRINCIPALE DE TRAITEMENT =============
# Désactivation du calcul des gradients pour économiser la mémoire
with torch.no_grad():
    for idx, (images, _) in enumerate(val_loader):
        # === Inférence du modèle ===
        # Transfert de l'image sur le device (GPU/CPU)
        image_tensor = images.to(DEVICE)
        
        # Propagation avant : génération des logits pour chaque classe
        outputs = model(image_tensor)
        
        # Conversion des logits en prédictions de classes
        # argmax sur la dimension des classes (dim=1)
        preds = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

        # === Récupération de l'identifiant de l'image ===
        # NOTE: Ceci fonctionne seulement si votre Dataset retourne le nom de fichier
        # Ajustez selon votre implémentation du dataset
        image_id = val_loader.dataset.image_info[idx]["file_name"]

        # === Extraction des masques spécifiques ===
        heart_mask = (preds == 1).astype(np.uint8)  # Masque binaire du cœur
        t4_mask = (preds == 8).astype(np.uint8)     # Masque binaire de T4

        # === Calcul manuel du VHS avec stockage des composantes ===
        # Vérification de la présence des structures nécessaires
        if heart_mask.sum() > 0 and t4_mask.sum() > 0:
            
            # --- Calcul des axes du cœur (identique à compute_vhs_from_mask) ---
            coords = np.column_stack(np.where(heart_mask > 0))
            mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
            center = mean[0]
            projections = (coords - center) @ eigenvectors.T
            heart_long_axis = projections[:, 0].max() - projections[:, 0].min()
            heart_short_axis = projections[:, 1].max() - projections[:, 1].min()

            # --- Calcul de la hauteur T4 ---
            contours, _ = cv2.findContours(t4_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            t4_contour = max(contours, key=cv2.contourArea)
            _, _, _, t4_height = cv2.boundingRect(t4_contour)

            # --- Validation et calcul final ---
            if t4_height > 0:
                # Calcul du VHS selon la formule standard
                vhs = (heart_long_axis + heart_short_axis) / t4_height
                
                # Stockage des résultats avec arrondi à 2 décimales
                results.append({
                    "image_id": image_id,
                    "vhs": round(vhs, 2),
                    "heart_long_axis": round(heart_long_axis, 2),
                    "heart_short_axis": round(heart_short_axis, 2),
                    "t4_height": round(t4_height, 2)
                })
                
                # Affichage de progression pour les valeurs calculées
                if idx % 10 == 0:  # Affichage tous les 10 images
                    print(f"  Image {idx+1}/{len(val_loader)} - VHS: {vhs:.2f}")
                    
            else:
                # Cas d'erreur : hauteur T4 nulle
                print(f"[!] Image {image_id} - T4 height zero. Skipping.")
        else:
            # Cas d'erreur : masques manquants
            print(f"[!] Image {idx} - Missing heart or T4 mask. Skipping.")

# ============= SAUVEGARDE DES RÉSULTATS =============
print(f"\n💾 Sauvegarde des résultats dans {output_file}...")

# Écriture du fichier CSV avec en-têtes
with open(output_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()  # Écriture des en-têtes de colonnes
    writer.writerows(results)  # Écriture de tous les résultats

# ============= RÉSUMÉ FINAL =============
print(f"\n✅ Traitement terminé !")
print(f"📊 Résultats sauvegardés : {len(results)} mesures VHS calculées")
print(f"📁 Fichier de sortie : {output_file}")

# Statistiques rapides si des résultats existent
if results:
    vhs_values = [r["vhs"] for r in results]
    print(f"📈 Statistiques VHS :")
    print(f"   • Moyenne : {np.mean(vhs_values):.2f}")
    print(f"   • Médiane : {np.median(vhs_values):.2f}")
    print(f"   • Min-Max : {min(vhs_values):.2f} - {max(vhs_values):.2f}")
else:
    print("⚠️  Aucun résultat calculé - vérifiez vos données et masques")