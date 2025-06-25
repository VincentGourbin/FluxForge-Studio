"""
Database migration script for MFLUX-Gradio.

This script migrates image generation history from an old MFLUX database
to the new MFLUX-Gradio database format, transforming the schema and
adding support for new features like ControlNet.

Usage:
    python migratedatabase.py

Note: Update the source_db and target_db paths before running.
"""

import sqlite3
import json
from datetime import datetime

def migrate_database():
    """
    Migrate image generation data from old MFLUX format to new MFLUX-Gradio format.
    
    Transforms:
    - Old 'requests' table to new 'images' table
    - LoRA paths and scales to JSON format
    - Adds ControlNet fields with default values
    - Validates and formats timestamps
    - Generates appropriate output filenames
    """
    # Database paths - UPDATE THESE BEFORE RUNNING
    source_db = 'path/to/your/source/request_history.db'  # Update this path
    target_db = './generated_images.db'

    # Connect to databases
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()

    try:
        # Select all data from source table
        source_cursor.execute('''
            SELECT prompt, model, seed, steps, height, width, guidance, timestamp,
                   quantize, path, lora_paths, lora_scales
            FROM requests
        ''')
        rows = source_cursor.fetchall()

        migrated_count = 0
        skipped_count = 0

        for row in rows:
            (prompt, model, seed, steps, height, width, guidance, timestamp,
             quantize, path, lora_paths, lora_scales) = row

            # Validate and convert timestamp
            if timestamp:
                try:
                    print(f"Processing timestamp: {timestamp[:19]}")
                    timestamp_dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f"Invalid timestamp for entry with prompt '{prompt}'. Entry skipped.")
                    skipped_count += 1
                    continue
            else:
                print(f"No timestamp for entry with prompt '{prompt}'. Entry skipped.")
                skipped_count += 1
                continue

            # Transform LoRA paths to JSON format
            if lora_paths:
                lora_paths_list = lora_paths.strip().split()
                # Prepend 'lora/' directory to each filename
                lora_paths_list = ['lora/' + filename for filename in lora_paths_list]
            else:
                lora_paths_list = []
            lora_paths_json = json.dumps(lora_paths_list)

            # Transform LoRA scales to JSON format
            if lora_scales:
                lora_scales_list = [float(scale) for scale in lora_scales.strip().split()]
            else:
                lora_scales_list = []
            lora_scales_json = json.dumps(lora_scales_list)

            # Generate output filename based on timestamp
            timestamp_formatted = timestamp_dt.strftime('%Y%m%d_%H%M%S')
            output_filename = f'outputimage/generated_image_{timestamp_formatted}.png'

            # Prepare values for insertion
            values = (
                timestamp,           # timestamp
                seed,                # seed
                prompt,              # prompt
                model,               # model_alias
                quantize,            # quantize
                steps,               # steps
                guidance,            # guidance
                height,              # height
                width,               # width
                path,                # path
                None,                # controlnet_image_path (default value)
                0.0,                 # controlnet_strength (default value)
                False,               # controlnet_save_canny (default value)
                lora_paths_json,     # lora_paths
                lora_scales_json,    # lora_scales
                output_filename      # output_filename
            )

            # Insert into target table
            target_cursor.execute('''
                INSERT INTO images (
                    timestamp, seed, prompt, model_alias, quantize, steps, guidance,
                    height, width, path, controlnet_image_path, controlnet_strength,
                    controlnet_save_canny, lora_paths, lora_scales, output_filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)

            migrated_count += 1

        # Commit changes and close connections
        target_conn.commit()
        
        print(f"Migration completed successfully!")
        print(f"Migrated: {migrated_count} entries")
        print(f"Skipped: {skipped_count} entries")

    except Exception as e:
        print(f"Migration failed with error: {e}")
        target_conn.rollback()
    finally:
        source_conn.close()
        target_conn.close()

if __name__ == "__main__":
    migrate_database()