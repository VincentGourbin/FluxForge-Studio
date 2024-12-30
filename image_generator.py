import os
import datetime
import random
import json
from pathlib import Path
from PIL import Image

import config
from database import save_image_info
from mflux import Flux1, Flux1Controlnet, Config, ConfigControlnet, ModelConfig
import gradio as gr

class ImageGenerator:
    def __init__(self):
        # Initialiser les variables d'état
        self.current_model_alias = None
        self.current_quantize = None
        self.current_path = None
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_model_type = None
        self.flux_model = None
        self.lora_data = config.lora_data
        self.lora_directory = config.lora_directory
        self.model_options = config.model_options
        self.quantize_options = config.quantize_options

    def generate_image(
        self,
        prompt,
        model_alias,
        quantize,
        steps,
        seed,
        metadata,
        guidance,
        height,
        width,
        path,
        controlnet_image_path,
        controlnet_strength,
        controlnet_save_canny,
        progress=gr.Progress(),
        *args
    ):
        # Création du dossier pour les images intermédiaires
        stepwise_dir = Path("stepwise_output")
        stepwise_dir.mkdir(parents=True, exist_ok=True)
        
        # Nettoyer le dossier des images précédentes
        for file in stepwise_dir.glob("*.png"):
            try:
                os.remove(file)
            except:
                pass

        # Traitement des paramètres
        if seed is None or int(seed) == 0:
            seed = random.randint(1, 2**32 - 1)
        else:
            seed = int(seed)
    
        height = int(height)
        width = int(width)
        steps = max(1, int(steps))  # Assurer que steps >= 1
        guidance = float(guidance) if guidance else 3.5
        if quantize == 'None' or quantize is None:
            quantize = None
        else:
            quantize = int(quantize)
        controlnet_strength = float(controlnet_strength)
        controlnet_save_canny = bool(controlnet_save_canny)
    
        # Traitement des sélections et des scales pour les LoRA
        num_lora = len(self.lora_data)
        lora_checkbox_values = args[:num_lora]
        lora_scale_values = args[num_lora:2*num_lora]
    
        lora_paths_list = []
        lora_scales_list = []
        for idx, (selected, scale) in enumerate(zip(lora_checkbox_values, lora_scale_values)):
            if selected:
                lora_info = self.lora_data[idx]
                lora_file = lora_info['file_name']
                lora_path = os.path.join(self.lora_directory, lora_file)
                lora_paths_list.append(lora_path)
                lora_scales_list.append(float(scale))
                # Ajouter le mot-clé d'activation au prompt
                prompt = f"{lora_info['activation_keyword']}, {prompt}"
    
        # Déterminer si ControlNet est utilisé
        use_controlnet = controlnet_image_path is not None
    
        # Vérifier si le modèle doit être rechargé
        if (model_alias != self.current_model_alias) or (quantize != self.current_quantize) or (path != self.current_path) or \
            (lora_paths_list != self.current_lora_paths) or (lora_scales_list != self.current_lora_scales) or (self.flux_model is None) or \
            (use_controlnet != (self.current_model_type == 'controlnet')):
    
            # Charger la configuration du modèle
            if path:
                model_config = ModelConfig.from_pretrained(path=path)
            else:
                model_config = ModelConfig.from_alias(alias=model_alias)
    
            # Charger le modèle approprié
            if use_controlnet:
                flux_model = Flux1Controlnet(
                    model_config=model_config,
                    quantize=quantize,
                    local_path=path,
                    lora_paths=lora_paths_list,
                    lora_scales=lora_scales_list,
                )
                current_model_type = 'controlnet'
            else:
                flux_model = Flux1(
                    model_config=model_config,
                    quantize=quantize,
                    local_path=path,
                    lora_paths=lora_paths_list,
                    lora_scales=lora_scales_list,
                )
                current_model_type = 'standard'
    
            # Mettre à jour les paramètres du modèle courant
            self.current_model_alias = model_alias
            self.current_quantize = quantize
            self.current_path = path
            self.current_lora_paths = lora_paths_list
            self.current_lora_scales = lora_scales_list
            self.current_model_type = current_model_type
            self.flux_model = flux_model
        else:
            # Si le modèle n'a pas changé, utilisez le modèle existant
            flux_model = self.flux_model
            current_model_type = self.current_model_type
    
        # Construire la configuration
        if use_controlnet:
            config_obj = ConfigControlnet(
                num_inference_steps=steps,
                guidance=guidance,
                height=height,
                width=width,
                controlnet_strength=controlnet_strength,
            )
        else:
            config_obj = Config(
                num_inference_steps=steps,
                guidance=guidance,
                height=height,
                width=width,
            )
    
        # Obtenir la date et l'heure actuelles
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
        # Adapter le nom du fichier de sortie pour éviter les caractères invalides
        output_dir = Path("outputimage")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
    
        # Générer l'image
        if use_controlnet:
            image = flux_model.generate_image(
                seed=seed,
                prompt=prompt,
                output=str(output_filename),
                controlnet_image_path=controlnet_image_path.name,
                controlnet_save_canny=controlnet_save_canny,
                config=config_obj,
                stepwise_output_dir=stepwise_dir
            )
        else:
            image = flux_model.generate_image(
                seed=seed,
                prompt=prompt,
                config=config_obj,
                stepwise_output_dir=stepwise_dir
            )
            
        image.save(path=str(output_filename), export_json_metadata=metadata)
    
        # Enregistrer les paramètres et le chemin de l'image dans la base de données
        save_image_info((
            timestamp_str,
            seed,
            prompt,
            model_alias,
            quantize,
            steps,
            guidance,
            height,
            width,
            path,
            controlnet_image_path.name if controlnet_image_path else None,
            controlnet_strength,
            controlnet_save_canny,
            json.dumps(lora_paths_list),  # Stocker les listes en tant que JSON
            json.dumps(lora_scales_list),
            str(output_filename)
        ))
    
        # Retourner l'image pour affichage dans Gradio
        return image.image  # Retourner l'image PIL

    def update_guidance_visibility(self, model_alias):
        if model_alias == "dev":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)