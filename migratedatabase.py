import sqlite3
import json
from datetime import datetime

# Chemins vers les bases de données
source_db = '/Users/vincent/Developpements/mflux/request_history.db'
target_db = '/Users/vincent/Developpements/mflux-gradio/generated_images.db'

# Connexion aux bases de données
source_conn = sqlite3.connect(source_db)
target_conn = sqlite3.connect(target_db)

source_cursor = source_conn.cursor()
target_cursor = target_conn.cursor()

# Sélectionner toutes les données de la table source
source_cursor.execute('''
    SELECT prompt, model, seed, steps, height, width, guidance, timestamp,
           quantize, path, lora_paths, lora_scales
    FROM requests
''')
rows = source_cursor.fetchall()

for row in rows:
    (prompt, model, seed, steps, height, width, guidance, timestamp,
     quantize, path, lora_paths, lora_scales) = row

    # Vérification et conversion du timestamp
    if timestamp:
        try:
            print(timestamp[:19])
            timestamp_dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Timestamp invalide pour l'entrée avec prompt '{prompt}'. Entrée ignorée.")
            continue  # Ignorer cette entrée et passer à la suivante
    else:
        print(f"Aucun timestamp pour l'entrée avec prompt '{prompt}'. Entrée ignorée.")
        continue  # Ignorer cette entrée et passer à la suivante

    # Transformation de lora_paths
    if lora_paths:
        lora_paths_list = lora_paths.strip().split()
        lora_paths_list = ['lora/' + filename for filename in lora_paths_list]
    else:
        lora_paths_list = []
    lora_paths_json = json.dumps(lora_paths_list)

    # Transformation de lora_scales
    if lora_scales:
        lora_scales_list = [float(scale) for scale in lora_scales.strip().split()]
    else:
        lora_scales_list = []
    lora_scales_json = json.dumps(lora_scales_list)

    # Transformation du timestamp pour output_filename
    timestamp_formatted = timestamp_dt.strftime('%Y%m%d_%H%M%S')
    output_filename = f'outputimage/generated_image_{timestamp_formatted}.png'

    # Préparation des valeurs à insérer
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
        None,                # controlnet_image_path (valeur par défaut)
        0.0,                 # controlnet_strength (valeur par défaut)
        False,               # controlnet_save_canny (valeur par défaut)
        lora_paths_json,     # lora_paths
        lora_scales_json,    # lora_scales
        output_filename      # output_filename
    )

    # Insertion dans la table cible
    target_cursor.execute('''
        INSERT INTO images (
            timestamp, seed, prompt, model_alias, quantize, steps, guidance,
            height, width, path, controlnet_image_path, controlnet_strength,
            controlnet_save_canny, lora_paths, lora_scales, output_filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', values)

# Valider les changements et fermer les connexions
target_conn.commit()
source_conn.close()
target_conn.close()

print("Migration terminée avec succès.")