# config.py
import os
# Configuration de l'environnement
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json
import torch


# Configuration de l'appareil (CPU, GPU, MPS)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Variables globales pour stocker le modèle chargé
current_model_alias = None
current_quantize = None
current_path = None
current_lora_paths = None
current_lora_scales = None
current_model_type = None  # 'standard' ou 'controlnet'
flux_model = None

# Chemin vers le dossier contenant les fichiers LoRA
lora_directory = 'lora'

# Chemin vers le fichier JSON contenant les informations des LoRA
lora_json_file = 'lora_info.json'

# Lire les informations des LoRA depuis le fichier JSON
with open(lora_json_file, 'r') as f:
    lora_data = json.load(f)

# Options de modèle et de quantification
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]

# Obtenir la liste des modèles disponibles avec Ollama
try:
    import ollama
    ollama_models = ollama.list()
    models_info = {}
    for model in ollama_models['models']:
        name = model['name']
        families = model['details'].get('families', [])
        models_info[name] = families
    model_names = list(models_info.keys())
except Exception as e:
    models_info = {}
    model_names = []
    print(f"Erreur lors de la récupération des modèles Ollama : {e}")