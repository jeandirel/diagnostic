"""
Générateur automatique de rapports de radiologie vétérinaire basé sur les scores VHS.

Ce module utilise l'intelligence artificielle (LLaMA 2 via Ollama) pour générer
des rapports diagnostiques à partir de mesures radiographiques thoraciques canines.
Le Vertebral Heart Score (VHS) est un indicateur clé de cardiomégalie chez les chiens.

Flux de traitement:
1. Chargement des données de mesures VHS depuis un fichier CSV
2. Génération de prompts structurés pour chaque image
3. Interrogation du modèle LLaMA 2 via l'API Ollama
4. Consolidation des rapports dans un fichier texte unique

Dépendances:
    - pandas: Manipulation des données tabulaires
    - ollama: Interface Python pour les modèles LLaMA locaux

Prérequis:
    - Serveur Ollama en fonctionnement avec le modèle 'llama2'
    - Fichier CSV 'vhs_results.csv' avec colonnes requises

Auteur: [À compléter]
Version: 1.0
Date: [À compléter]
"""

import pandas as pd
import ollama

# === Chargement des données source ===
# Lecture du dataset contenant les mesures radiographiques
# Structure attendue: image_id, vhs, heart_long_axis, heart_short_axis, t4_height
df = pd.read_csv("vhs_results.csv")

# === Configuration des sorties ===
# Fichier de consolidation pour tous les rapports générés
output_file = "vhs_llama2_reports.txt"

# Liste d'accumulation des rapports formatés
all_reports = []

# === Génération séquentielle des rapports ===
# Traitement ligne par ligne du dataset avec suivi de progression
for idx, row in df.iterrows():
    # === Construction du prompt médical structuré ===
    # Template de prompt optimisé pour la radiologie vétérinaire
    # Inclut le contexte professionnel et les paramètres de mesure
    prompt = f"""
You are a veterinary radiologist. Based on the following measurements, write a short diagnostic report:

- Image ID: {row['image_id']}
- Vertebral Heart Score (VHS): {row['vhs']}
- Heart Long Axis: {row['heart_long_axis']} pixels
- Heart Short Axis: {row['heart_short_axis']} pixels
- T4 Vertebral Height: {row['t4_height']} pixels

Describe whether the VHS is normal, mildly elevated, or significantly elevated. Mention if further diagnostics are advised.
"""

    # Affichage du statut de progression avec emoji médical
    print(f"🩺 Generating report for: {row['image_id']}")

    # === Appel au modèle d'IA via Ollama ===
    # Utilisation de l'API Ollama pour interroger LLaMA 2 en local
    # Format conversationnel avec rôle utilisateur défini
    response = ollama.chat(
        model='llama2',                                    # Modèle LLaMA 2 pré-chargé
        messages=[{"role": "user", "content": prompt}]     # Structure de conversation
    )

    # === Extraction et formatage de la réponse ===
    # Récupération du contenu textuel de la réponse IA
    report = response['message']['content']
    
    # Formatage standardisé avec délimiteurs pour parsing ultérieur
    # Structure: En-tête d'image + Contenu du rapport + Séparateur visuel
    report_block = f"Image: {row['image_id']}\n{report}\n{'-'*60}\n"
    
    # Accumulation dans la liste des rapports
    all_reports.append(report_block)

# === Sauvegarde consolidée ===
# Écriture de tous les rapports dans un fichier texte unique
# Mode 'w' pour écrasement complet (évite les duplications)
with open(output_file, "w") as f:
    f.writelines(all_reports)

# === Confirmation de fin de traitement ===
print(f"\n All reports saved to {output_file}")