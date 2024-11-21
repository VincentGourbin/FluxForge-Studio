# database.py
import sqlite3
import json
import os
from pathlib import Path

# Chemin vers la base de données
db_path = 'generated_images.db'

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Création de la table pour stocker les images et les paramètres
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            seed INTEGER,
            prompt TEXT,
            model_alias TEXT,
            quantize INTEGER,
            steps INTEGER,
            guidance REAL,
            height INTEGER,
            width INTEGER,
            path TEXT,
            controlnet_image_path TEXT,
            controlnet_strength REAL,
            controlnet_save_canny BOOLEAN,
            lora_paths TEXT,
            lora_scales TEXT,
            output_filename TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_image_info(details):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (
            timestamp, seed, prompt, model_alias, quantize, steps, guidance,
            height, width, path, controlnet_image_path, controlnet_strength,
            controlnet_save_canny, lora_paths, lora_scales, output_filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', details)
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, output_filename FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()

    images = []
    for record in records:
        image_id, output_filename = record
        if os.path.exists(output_filename):
            images.append(output_filename)
    return images

def get_image_details(index):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    if 0 <= index < len(records):
        record = records[index]
        details = dict(zip(columns, record))
        # Formater les détails pour affichage
        details_text = '\n'.join([f"{key}: {value}" for key, value in details.items()])
        return details_text
    else:
        return "Aucune information disponible pour cette image."

def delete_image(selected_image_index):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    if 0 <= selected_image_index < len(records):
        record = records[selected_image_index]
        image_id = record[columns.index('id')]
        output_filename = record[columns.index('output_filename')]

        # Supprimer le fichier image du système de fichiers
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Supprimer l'enregistrement de la base de données
        cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
        conn.close()

        return True, "Image supprimée avec succès."
    else:
        conn.close()
        return False, "Aucune image correspondante trouvée."