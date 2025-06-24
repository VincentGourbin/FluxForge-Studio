# database.py
"""
Database module for managing image generation history and metadata.
Handles SQLite operations for storing and retrieving generated images information.
"""

import sqlite3
import json
import os
from pathlib import Path

# Database file path
db_path = 'generated_images.db'

def init_db():
    """
    Initialize the SQLite database with the required table structure.
    Creates the 'images' table if it doesn't exist, with all necessary columns 
    to store image generation parameters and metadata.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create table to store images and their generation parameters
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
    """
    Save image generation information to the database.
    
    Args:
        details (tuple): A tuple containing all the image generation parameters
                        in the order matching the database columns (excluding id)
    """
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
    """
    Load the history of generated images from the database.
    Only returns images that still exist on the filesystem.
    
    Returns:
        list: List of image file paths for existing generated images,
              ordered by timestamp (newest first)
    """
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
    """
    Retrieve detailed information about a specific image by its index.
    
    Args:
        index (int): Index of the image in the history (0-based)
        
    Returns:
        str: Formatted string containing all image generation parameters,
             or error message if index is invalid
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    if 0 <= index < len(records):
        record = records[index]
        details = dict(zip(columns, record))
        # Format details for display
        details_text = '\n'.join([f"{key}: {value}" for key, value in details.items()])
        return details_text
    else:
        return "Aucune information disponible pour cette image."

def delete_image(selected_image_index):
    """
    Delete an image and its database record by index.
    Removes both the image file from the filesystem and the database entry.
    
    Args:
        selected_image_index (int): Index of the image to delete (0-based)
        
    Returns:
        tuple: (success: bool, message: str) indicating operation result
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    if 0 <= selected_image_index < len(records):
        record = records[selected_image_index]
        image_id = record[columns.index('id')]
        output_filename = record[columns.index('output_filename')]

        # Remove image file from filesystem
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Remove database record
        cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
        conn.close()

        return True, "Image supprimée avec succès."
    else:
        conn.close()
        return False, "Aucune image correspondante trouvée."