import cv2
import torch
import torchvision.transforms as T
import numpy as np
import os
import sys
from scipy.ndimage import binary_opening, binary_closing, label

# Obtener nombre del video de los argumentos
if len(sys.argv) > 1:
    video_name = sys.argv[1]
else:
    print("❌ Error: No se proporcionó el nombre del video.")
    exit()

# Rutas relativas
video_path = os.path.join("data", "Videos", video_name)
output_folder = os.path.join("data", "VideosResNet50")
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, f"{video_name.replace('.avi', '')}_resnet.avi")

# === Modelo (ResNet50 backbone) ===
model = torch.hub.load("pytorch/vision", "deeplabv3_resnet50", pretrained=True).eval()

# === Transformación ===
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# === Procesar video ===
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        probs = torch.softmax(output, dim=0)[1]

    mask = (probs > 0.5).byte().cpu().numpy()
    mask = binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)

    labeled_mask, num_features = label(mask)
    refined_mask = np.zeros_like(mask)
    for region_label in range(1, num_features + 1):
        region = (labeled_mask == region_label)
        if region.sum() > 100:
            refined_mask[region] = 1

    # Redimensionar la máscara refinada
    refined_mask_resized = cv2.resize(refined_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    # Crear máscara roja pura
    red_mask = np.zeros_like(frame)
    red_mask[:, :, 2] = 255

    # Reemplazar píxeles donde la máscara == 1 con rojo puro
    frame_segmented = np.where(refined_mask_resized[:, :, None] == 1, red_mask, frame)

    out.write(frame_segmented)

    frame_number += 1
    if frame_number % 50 == 0:
        print(f"Procesados {frame_number} frames...")

print("✅ Video final (ResNet50) generado correctamente.")
print(f"Guardado en: {output_video_path}")
cap.release()
out.release()
