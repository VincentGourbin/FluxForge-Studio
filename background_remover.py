# background_remover.py

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
from config import device

# Chargement du modèle
def load_background_removal_model():
    modelbgrm = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    modelbgrm.to(device)
    modelbgrm.eval()
    return modelbgrm

# Définition des transformations d'image
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fonction pour supprimer l'arrière-plan
def remove_background(input_image, modelbgrm):
    image = input_image.convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = modelbgrm(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    return image