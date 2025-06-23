"""
Module de génération de rapports PDF à partir d'un fichier texte consolidé.

Ce script traite un fichier texte contenant plusieurs rapports délimités par des balises "Image:"
et génère un fichier PDF individuel pour chaque rapport identifié.

Dépendances:
    - reportlab: Bibliothèque de génération PDF
    - os: Gestion du système de fichiers

Auteur: [À compléter]
Version: 1.0
Date: [À compléter]
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# === Configuration des chemins ===
# Chemin vers le fichier texte consolidé contenant tous les rapports
report_txt_path = "vhs_llama2_reports.txt"

# Répertoire de sortie pour les fichiers PDF générés
output_dir = "generated_reports/pdf"

# Création du dossier de sortie s'il n'existe pas
# exist_ok=True évite les erreurs si le dossier existe déjà
os.makedirs(output_dir, exist_ok=True)

# === Chargement et analyse du fichier texte source ===
# Lecture complète du fichier en mode texte avec encodage par défaut
with open(report_txt_path, "r") as f:
    lines = f.readlines()

# Variables de contrôle pour le parsing séquentiel
current_filename = None          # Nom du fichier PDF en cours de traitement
current_report_lines = []        # Lignes du rapport en cours d'accumulation

def save_pdf(filename, content_lines):
    """
    Génère un fichier PDF à partir d'un nom de fichier et de lignes de contenu.
    
    Cette fonction crée un document PDF avec mise en page basique:
    - Format A4
    - Police Helvetica 11pt
    - Marges de 40pt à gauche et 50pt en haut
    - Interligne de 18pt
    - Gestion automatique des sauts de page
    
    Args:
        filename (str): Nom du fichier PDF à créer (sans extension)
        content_lines (list): Liste des lignes de texte à inclure dans le PDF
    
    Returns:
        None
    
    Side Effects:
        Crée un fichier PDF dans le répertoire output_dir
    """
    # Construction du chemin complet avec extension PDF
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    
    # Initialisation du canvas ReportLab avec format A4
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Position verticale initiale (coordonnées ReportLab: origine en bas à gauche)
    y = height - 50  # Marge supérieure de 50 points
    
    # Configuration de la police par défaut
    c.setFont("Helvetica", 11)

    # Traitement séquentiel de chaque ligne de contenu
    for line in content_lines:
        # Vérification de l'espace disponible sur la page courante
        if y < 40:  # Marge inférieure de sécurité
            # Saut de page et réinitialisation
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)  # Réapplication de la police après saut de page
        
        # Insertion de la ligne avec suppression des espaces en début/fin
        c.drawString(40, y, line.strip())  # Marge gauche de 40 points
        
        # Déplacement vers la ligne suivante
        y -= 18  # Interligne de 18 points

    # Finalisation et sauvegarde du document PDF
    c.save()

# === Traitement principal: segmentation des rapports ===
# Parcours séquentiel du fichier pour identifier les délimiteurs de rapports
for line in lines:
    # Détection d'un nouveau rapport via la balise "Image:"
    if line.startswith("Image:"):
        # Sauvegarde du rapport précédent s'il existe
        if current_filename and current_report_lines:
            save_pdf(current_filename, current_report_lines)
        
        # Extraction et normalisation du nom de fichier
        # Suppression de "Image:" et des extensions d'image courantes
        current_filename = line.split("Image:")[1].strip().replace(".jpg", "").replace(".png", "")
        
        # Initialisation du nouveau rapport avec la ligne de délimitation
        current_report_lines = [line]
    else:
        # Accumulation des lignes du rapport en cours
        current_report_lines.append(line)

# === Traitement du dernier rapport ===
# Gestion du cas où le fichier ne se termine pas par un nouveau délimiteur
if current_filename and current_report_lines:
    save_pdf(current_filename, current_report_lines)

# === Confirmation de fin de traitement ===
print(f"PDF reports saved to {output_dir}")