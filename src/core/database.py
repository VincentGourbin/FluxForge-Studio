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

def check_and_migrate_database():
    """
    Check database version and perform migration if needed.
    Migrates from old column-based structure to new JSON-based structure.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if we have the new structure
    cursor.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]
    
    has_metadata_json = 'metadata_json' in columns
    has_old_structure = 'quantize' in columns or 'steps' in columns
    
    if not has_metadata_json and has_old_structure:
        migrate_to_json_structure(conn)
    elif not has_metadata_json:
        create_new_structure(conn)
    
    conn.close()

def migrate_to_json_structure(conn):
    """
    Migrate existing database from column-based to JSON-based structure.
    """
    cursor = conn.cursor()
    
    # Read all existing data
    cursor.execute("SELECT * FROM images")
    existing_data = cursor.fetchall()
    cursor.execute("PRAGMA table_info(images)")
    old_columns = [column[1] for column in cursor.fetchall()]
    
    # Create new table structure
    cursor.execute('''
        CREATE TABLE images_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            generation_type TEXT NOT NULL,
            output_filename TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
    ''')
    
    # Migrate data
    for row in existing_data:
        row_dict = dict(zip(old_columns, row))
        
        # Determine generation type
        if row_dict.get('controlnet_image_path'):
            generation_type = 'controlnet'
        elif 'flux_fill' in str(row_dict.get('model_alias', '')).lower():
            generation_type = 'flux_fill'
        elif 'kontext' in str(row_dict.get('model_alias', '')).lower():
            generation_type = 'kontext'
        elif 'upscaler' in str(row_dict.get('model_alias', '')).lower():
            generation_type = 'upscaler'
        else:
            generation_type = 'standard'
        
        # Create metadata JSON
        metadata = {
            'seed': row_dict.get('seed', 0),
            'prompt': row_dict.get('prompt', ''),
            'model_alias': row_dict.get('model_alias', ''),
            'steps': row_dict.get('steps', 0),
            'guidance': row_dict.get('guidance', 0.0),
            'height': row_dict.get('height', 0),
            'width': row_dict.get('width', 0),
            'lora_paths': json.loads(row_dict.get('lora_paths', '[]')),
            'lora_scales': json.loads(row_dict.get('lora_scales', '[]')),
        }
        
        # Add specific metadata based on generation type
        if generation_type == 'controlnet':
            metadata['controlnet'] = {
                'type': row_dict.get('quantize', 'None'),  # quantize was repurposed
                'image_path': row_dict.get('controlnet_image_path'),
                'strength': row_dict.get('controlnet_strength', 1.0),
                'save_canny': row_dict.get('controlnet_save_canny', False)
            }
        elif generation_type == 'upscaler':
            metadata['upscaler'] = {
                'multiplier': row_dict.get('controlnet_strength', 2.0)  # repurposed field
            }
        
        cursor.execute('''
            INSERT INTO images_new (timestamp, generation_type, output_filename, metadata_json)
            VALUES (?, ?, ?, ?)
        ''', (
            row_dict.get('timestamp', ''),
            generation_type,
            row_dict.get('output_filename', ''),
            json.dumps(metadata)
        ))
    
    # Replace old table
    cursor.execute("DROP TABLE images")
    cursor.execute("ALTER TABLE images_new RENAME TO images")
    conn.commit()

def create_new_structure(conn):
    """
    Create new JSON-based database structure.
    """
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            generation_type TEXT NOT NULL,
            output_filename TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
    ''')
    conn.commit()

def init_db():
    """
    Initialize the SQLite database with migration support.
    """
    check_and_migrate_database()
    init_lora_table()

def save_image_info(details):
    """
    Legacy function for backward compatibility. Use specific save functions instead.
    """
    # For backward compatibility during transition
    # This will be phased out once all modules use the new functions
    pass

def save_standard_generation(timestamp, seed, prompt, model_alias, steps, guidance, 
                           height, width, lora_paths, lora_scales, output_filename, negative_prompt=None,
                           total_generation_time=None, model_generation_time=None):
    """
    Save standard text-to-image generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': model_alias,
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales
    }
    
    # Add negative_prompt if provided (for Qwen-Image)
    if negative_prompt:
        metadata['negative_prompt'] = negative_prompt
    
    # Add generation timing information if provided
    if total_generation_time is not None:
        metadata['total_generation_time'] = round(total_generation_time, 2)
    if model_generation_time is not None:
        metadata['model_generation_time'] = round(model_generation_time, 2)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'standard', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_flux_fill_generation(timestamp, seed, prompt, mode, steps, guidance, 
                            height, width, lora_paths, lora_scales, output_filename):
    """
    Save FLUX Fill (inpainting/outpainting) generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': 'flux_fill',
        'mode': mode,  # 'Inpainting' or 'Outpainting'
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'flux_fill', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_kontext_generation(timestamp, seed, prompt, steps, guidance, 
                          height, width, lora_paths, lora_scales, output_filename):
    """
    Save Kontext (image editing) generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': 'kontext',
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'kontext', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_upscaler_generation(timestamp, seed, multiplier, height, width, output_filename):
    """
    Save upscaler generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': f'Image upscaling x{multiplier}',
        'model_alias': 'upscaler',
        'multiplier': multiplier,
        'steps': 28,
        'guidance': 3.5,
        'height': height,
        'width': width,
        'lora_paths': [],
        'lora_scales': []
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'upscaler', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_background_removal_generation(timestamp, seed, height, width, output_filename):
    """
    Save background removal generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': 'Background removal processing',
        'model_alias': 'RMBG-2.0',
        'steps': 1,
        'guidance': 1.0,
        'height': height,
        'width': width,
        'lora_paths': [],
        'lora_scales': []
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'background_removal', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_controlnet_generation(timestamp, seed, prompt, model_alias, controlnet_type,
                             controlnet_strength, steps, guidance, height, width, 
                             lora_paths, lora_scales, output_filename):
    """
    Save ControlNet generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': model_alias,
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales,
        'controlnet': {
            'type': controlnet_type,
            'strength': controlnet_strength
        }
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'controlnet', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_flux_depth_generation(timestamp, seed, prompt, steps, guidance, 
                              height, width, lora_paths, lora_scales, output_filename):
    """
    Save FLUX Depth generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': 'flux_depth',
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales,
        'depth_control': True
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'flux_depth', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_flux_canny_generation(timestamp, seed, prompt, steps, guidance, 
                              low_threshold, high_threshold, height, width, 
                              lora_paths, lora_scales, output_filename):
    """
    Save FLUX Canny generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': prompt,
        'model_alias': 'flux_canny',
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': lora_paths,
        'lora_scales': lora_scales,
        'canny_control': True,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'flux_canny', output_filename, json.dumps(metadata)))
    conn.commit()
    conn.close()

def save_flux_redux_generation(timestamp, seed, guidance, steps, variation_strength, 
                              height, width, output_filename):
    """
    Save FLUX Redux generation to database.
    """
    metadata = {
        'seed': seed,
        'prompt': 'Image variation/refinement',
        'model_alias': 'flux_redux',
        'steps': steps,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_paths': [],
        'lora_scales': [],
        'redux_variation': True,
        'variation_strength': variation_strength
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (timestamp, generation_type, output_filename, metadata_json)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, 'flux_redux', output_filename, json.dumps(metadata)))
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
    cursor.execute('SELECT output_filename FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()

    images = []
    for record in records:
        output_filename = record[0]
        if os.path.exists(output_filename):
            images.append(output_filename)
    return images

def get_image_details(index):
    """
    Retrieve detailed information about a specific image by its index.
    Only considers images that still exist on the filesystem.
    
    Args:
        index (int): Index of the image in the history (0-based)
        
    Returns:
        str: Formatted string containing all image generation parameters,
             or error message if index is invalid
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, generation_type, output_filename, metadata_json FROM images ORDER BY timestamp DESC')
    all_records = cursor.fetchall()
    conn.close()

    # Filter to only include records where the file still exists (same logic as load_history)
    existing_records = []
    for record in all_records:
        timestamp, generation_type, output_filename, metadata_json = record
        if os.path.exists(output_filename):
            existing_records.append(record)

    if 0 <= index < len(existing_records):
        timestamp, generation_type, output_filename, metadata_json = existing_records[index]
        
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return "Error: Invalid metadata format"
        
        # Format details for display
        details_lines = [
            f"🕒 Timestamp: {timestamp}",
            f"🎨 Generation Type: {generation_type.title()}",
            f"📁 File: {os.path.basename(output_filename)}",
            "",
            "📊 Generation Parameters:",
        ]
        
        # Basic parameters
        if 'prompt' in metadata:
            details_lines.append(f"  💬 Prompt: {metadata['prompt']}")
        if 'negative_prompt' in metadata and metadata['negative_prompt']:
            details_lines.append(f"  🚫 Negative Prompt: {metadata['negative_prompt']}")
        if 'seed' in metadata:
            details_lines.append(f"  🎲 Seed: {metadata['seed']}")
        if 'model_alias' in metadata:
            details_lines.append(f"  🤖 Model: {metadata['model_alias']}")
        if 'steps' in metadata:
            details_lines.append(f"  🔄 Steps: {metadata['steps']}")
        if 'guidance' in metadata:
            details_lines.append(f"  🎚️ Guidance: {metadata['guidance']}")
        if 'height' in metadata and 'width' in metadata:
            details_lines.append(f"  📐 Size: {metadata['width']}x{metadata['height']}")
        
        # Type-specific parameters
        if generation_type == 'flux_fill' and 'mode' in metadata:
            details_lines.append(f"  🖌️ Mode: {metadata['mode']}")
        elif generation_type == 'upscaler' and 'multiplier' in metadata:
            details_lines.append(f"  📈 Multiplier: {metadata['multiplier']}x")
        elif generation_type == 'flux_depth':
            details_lines.append(f"  🌊 Type: Depth-guided generation")
            if 'depth_control' in metadata:
                details_lines.append(f"  🎛️ Depth Control: {metadata['depth_control']}")
        elif generation_type == 'flux_canny':
            details_lines.append(f"  🖋️ Type: Canny edge-guided generation")
            if 'canny_control' in metadata:
                details_lines.append(f"  🎛️ Canny Control: {metadata['canny_control']}")
            if 'low_threshold' in metadata and 'high_threshold' in metadata:
                details_lines.append(f"  📉 Low Threshold: {metadata['low_threshold']}")
                details_lines.append(f"  📈 High Threshold: {metadata['high_threshold']}")
        elif generation_type == 'flux_redux':
            details_lines.append(f"  🔄 Type: Image variation/refinement")
            if 'redux_variation' in metadata:
                details_lines.append(f"  🎛️ Redux Variation: {metadata['redux_variation']}")
            if 'variation_strength' in metadata:
                details_lines.append(f"  💫 Variation Strength: {metadata['variation_strength']}")
        elif generation_type == 'controlnet' and 'controlnet' in metadata:
            cn_data = metadata['controlnet']
            details_lines.append(f"  🎛️ ControlNet Type: {cn_data.get('type', 'Unknown')}")
            details_lines.append(f"  🎛️ ControlNet Strength: {cn_data.get('strength', 1.0)}")
        
        # LoRA information
        if 'lora_paths' in metadata and 'lora_scales' in metadata:
            lora_paths = metadata['lora_paths']
            lora_scales = metadata['lora_scales']
            if lora_paths and len(lora_paths) > 0:
                details_lines.append("")
                details_lines.append("🎨 LoRA Models:")
                for i, (path, scale) in enumerate(zip(lora_paths, lora_scales)):
                    lora_name = os.path.basename(path)
                    details_lines.append(f"  {i+1}. {lora_name} (scale: {scale})")
        
        # Raw JSON (for debugging)
        details_lines.extend([
            "",
            "🔧 Raw Metadata JSON:",
            json.dumps(metadata, indent=2)
        ])
        
        return '\n'.join(details_lines)
    else:
        return "No information available for this image."

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

        return True, "Image deleted successfully."
    else:
        conn.close()
        return False, "No matching image found."

def sync_gallery_and_disk():
    """
    Synchronise la base de données, la gallery et les fichiers sur le disque.
    
    Cette fonction :
    1. Trouve les images sur le disque qui ne sont pas dans la base de données
       et les déplace vers orphaned_pictures/
    2. Trouve les entrées en base qui pointent vers des fichiers inexistants
       et les supprime de la base de données
    3. Gère les fichiers JSON de métadonnées associés
    
    Returns:
        list: Updated gallery images list
    """
    import glob
    import shutil
    from pathlib import Path
    
    try:
        # Créer le dossier orphaned_pictures s'il n'existe pas
        orphaned_dir = Path("orphaned_pictures")
        orphaned_dir.mkdir(exist_ok=True)
        
        # Récupérer toutes les entrées de la base de données
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT output_filename FROM images')
        db_files = {record[0] for record in cursor.fetchall()}
        conn.close()
        
        # Récupérer tous les fichiers d'images dans outputimage/
        output_dir = Path("outputimage")
        if not output_dir.exists():
            return False, "Le dossier outputimage/ n'existe pas.", {}
            
        # Patterns pour les images et JSON
        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        disk_images = set()
        disk_jsons = set()
        
        for pattern in image_patterns:
            disk_images.update(output_dir.glob(pattern))
        
        # Récupérer tous les fichiers JSON
        disk_jsons.update(output_dir.glob("*.json"))
        
        # Convertir en chemins relatifs string pour comparaison
        disk_images_str = {str(img) for img in disk_images}
        disk_jsons_str = {str(json_file) for json_file in disk_jsons}
        
        # Statistiques
        stats = {
            "images_in_db": len(db_files),
            "images_on_disk": len(disk_images_str),
            "orphaned_moved": 0,
            "db_entries_removed": 0,
            "jsons_moved": 0,
            "orphaned_jsons_moved": 0
        }
        
        # 1. Trouver les images sur le disque mais pas en base (orphelines)
        orphaned_images = disk_images_str - db_files
        
        for orphaned_image in orphaned_images:
            orphaned_path = Path(orphaned_image)
            if orphaned_path.exists():
                # Déplacer l'image
                dest_image = orphaned_dir / orphaned_path.name
                shutil.move(str(orphaned_path), str(dest_image))
                stats["orphaned_moved"] += 1
                
                # Chercher et déplacer le JSON associé s'il existe
                json_path = orphaned_path.with_suffix('.json')
                if json_path.exists():
                    dest_json = orphaned_dir / json_path.name
                    shutil.move(str(json_path), str(dest_json))
                    stats["jsons_moved"] += 1
        
        # 2. Trouver les JSON orphelins (JSON sans image associée)
        orphaned_jsons = set()
        for json_file in disk_jsons:
            json_path = Path(json_file)
            # Chercher l'image correspondante (même nom, différentes extensions)
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
            corresponding_image_found = False
            
            for ext in image_extensions:
                corresponding_image = json_path.with_suffix(ext)
                if corresponding_image in disk_images:
                    corresponding_image_found = True
                    break
            
            if not corresponding_image_found:
                orphaned_jsons.add(json_file)
        
        # Déplacer les JSON orphelins
        for orphaned_json in orphaned_jsons:
            json_path = Path(orphaned_json)
            if json_path.exists():
                dest_json = orphaned_dir / json_path.name
                shutil.move(str(json_path), str(dest_json))
                stats["orphaned_jsons_moved"] += 1
        
        # 3. Trouver les entrées en base qui pointent vers des fichiers inexistants
        missing_files = db_files - disk_images_str
        
        if missing_files:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for missing_file in missing_files:
                cursor.execute('DELETE FROM images WHERE output_filename = ?', (missing_file,))
                stats["db_entries_removed"] += 1
            
            conn.commit()
            conn.close()
        
        # Générer le message de résultat
        message_parts = []
        if stats["orphaned_moved"] > 0:
            message_parts.append(f"🗂️ {stats['orphaned_moved']} orphaned image(s) moved")
        if stats["jsons_moved"] > 0:
            message_parts.append(f"📄 {stats['jsons_moved']} associated JSON file(s) moved")
        if stats["orphaned_jsons_moved"] > 0:
            message_parts.append(f"📄 {stats['orphaned_jsons_moved']} orphaned JSON file(s) moved")
        if stats["db_entries_removed"] > 0:
            message_parts.append(f"🗑️ {stats['db_entries_removed']} database entry(ies) removed")
        
        if not message_parts:
            message = "Gallery and disk already synchronized - no action required"
        else:
            message = "Synchronization completed: " + ", ".join(message_parts)
        
        # Return updated gallery list
        return load_history()
        
    except Exception as e:
        return load_history()  # Return current gallery even if sync failed

# ==============================================================================
# LORA MANAGEMENT FUNCTIONS
# ==============================================================================

def init_lora_table():
    """
    Initialize the LoRA table in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            activation_keyword TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            file_size INTEGER,
            is_active INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

def get_all_lora():
    """
    Get all active LoRA models from the database.
    
    Returns:
        list: List of LoRA dictionaries with all information
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, file_name, description, activation_keyword, created_at, updated_at, file_size, is_active
        FROM lora 
        WHERE is_active = 1
        ORDER BY file_name COLLATE NOCASE
    ''')
    
    lora_list = []
    for row in cursor.fetchall():
        lora_list.append({
            'id': row[0],
            'file_name': row[1],
            'description': row[2],
            'activation_keyword': row[3],
            'created_at': row[4],
            'updated_at': row[5],
            'file_size': row[6],
            'is_active': row[7]
        })
    
    conn.close()
    return lora_list

def get_lora_by_id(lora_id):
    """
    Get a specific LoRA by its ID.
    
    Args:
        lora_id (int): The ID of the LoRA
        
    Returns:
        dict: LoRA information or None if not found
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, file_name, description, activation_keyword, created_at, updated_at, file_size, is_active
        FROM lora 
        WHERE id = ?
    ''', (lora_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'file_name': row[1],
            'description': row[2],
            'activation_keyword': row[3],
            'created_at': row[4],
            'updated_at': row[5],
            'file_size': row[6],
            'is_active': row[7]
        }
    return None

def add_lora(file_name, description, activation_keyword=""):
    """
    Add a new LoRA to the database.
    
    Args:
        file_name (str): Name of the LoRA file
        description (str): Description of the LoRA
        activation_keyword (str): Activation keyword for the LoRA
        
    Returns:
        tuple: (success: bool, message: str, lora_id: int)
    """
    from datetime import datetime
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if file already exists
        cursor.execute("SELECT id FROM lora WHERE file_name = ?", (file_name,))
        if cursor.fetchone():
            conn.close()
            return False, f"LoRA with filename '{file_name}' already exists", None
        
        # Get file size if file exists
        lora_file_path = Path('lora') / file_name
        file_size = None
        if lora_file_path.exists():
            file_size = lora_file_path.stat().st_size
        
        current_time = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO lora (file_name, description, activation_keyword, created_at, updated_at, file_size, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_name, description, activation_keyword, current_time, current_time, file_size, 1))
        
        lora_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return True, f"LoRA '{file_name}' added successfully", lora_id
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Error adding LoRA: {str(e)}", None

def update_lora(lora_id, description, activation_keyword):
    """
    Update an existing LoRA in the database.
    
    Args:
        lora_id (int): ID of the LoRA to update
        description (str): New description
        activation_keyword (str): New activation keyword
        
    Returns:
        tuple: (success: bool, message: str)
    """
    from datetime import datetime
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if LoRA exists
        cursor.execute("SELECT file_name FROM lora WHERE id = ?", (lora_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False, f"LoRA with ID {lora_id} not found"
        
        file_name = result[0]
        current_time = datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE lora 
            SET description = ?, activation_keyword = ?, updated_at = ?
            WHERE id = ?
        ''', (description, activation_keyword, current_time, lora_id))
        
        conn.commit()
        conn.close()
        
        return True, f"LoRA '{file_name}' updated successfully"
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Error updating LoRA: {str(e)}"

def delete_lora(lora_id, delete_file=False):
    """
    Delete a LoRA from the database and optionally from disk.
    
    Args:
        lora_id (int): ID of the LoRA to delete
        delete_file (bool): Whether to also delete the file from disk
        
    Returns:
        tuple: (success: bool, message: str)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get LoRA info before deletion
        cursor.execute("SELECT file_name FROM lora WHERE id = ?", (lora_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False, f"LoRA with ID {lora_id} not found"
        
        file_name = result[0]
        
        # Delete from database
        cursor.execute("DELETE FROM lora WHERE id = ?", (lora_id,))
        
        # Delete file if requested
        if delete_file:
            lora_file_path = Path('lora') / file_name
            if lora_file_path.exists():
                lora_file_path.unlink()
                message = f"LoRA '{file_name}' deleted from database and disk"
            else:
                message = f"LoRA '{file_name}' deleted from database (file not found on disk)"
        else:
            message = f"LoRA '{file_name}' deleted from database"
        
        conn.commit()
        conn.close()
        
        return True, message
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Error deleting LoRA: {str(e)}"

def get_lora_for_image_generator():
    """
    Get LoRA data in the format expected by the image generator.
    
    Returns:
        list: List of LoRA dictionaries compatible with existing image generator
    """
    lora_list = get_all_lora()
    
    # Convert to format expected by image generator
    formatted_lora = []
    for lora in lora_list:
        formatted_lora.append({
            'file_name': lora['file_name'],
            'description': lora['description'],
            'activation_keyword': lora['activation_keyword'] or ""
        })
    
    return formatted_lora

def refresh_lora_file_sizes():
    """
    Refresh file sizes for all LoRA entries in the database.
    This is useful when files are added/modified outside the application.
    
    Returns:
        tuple: (success: bool, message: str, updated_count: int)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, file_name FROM lora")
        lora_entries = cursor.fetchall()
        
        updated_count = 0
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        for lora_id, file_name in lora_entries:
            lora_file_path = Path('lora') / file_name
            
            if lora_file_path.exists():
                file_size = lora_file_path.stat().st_size
                cursor.execute('''
                    UPDATE lora 
                    SET file_size = ?, updated_at = ?
                    WHERE id = ?
                ''', (file_size, current_time, lora_id))
                updated_count += 1
            else:
                # File doesn't exist, set size to None
                cursor.execute('''
                    UPDATE lora 
                    SET file_size = NULL, updated_at = ?
                    WHERE id = ?
                ''', (current_time, lora_id))
        
        conn.commit()
        conn.close()
        
        return True, f"File sizes updated for {updated_count} LoRA entries", updated_count
        
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Error refreshing file sizes: {str(e)}", 0