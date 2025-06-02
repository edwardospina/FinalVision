import cv2
import pandas as pd
import numpy as np
import os
import sys

# Obtener nombre del video
if len(sys.argv) > 1:
    video_name = sys.argv[1]
else:
    print("❌ Error: No se proporcionó el nombre del video.")
    exit()

# === Rutas relativas ===
volume_tracings_csv = os.path.join("utils", "VolumeTracings.csv")
video_original = os.path.join("data", "Videos", video_name)
video_resnet = os.path.join("data", "VideosResNet50", f"{video_name.replace('.avi', '')}_resnet.avi")
video_mobilenet = os.path.join("data", "VideosMobileNetV3", f"{video_name.replace('.avi', '')}_mobilenet.avi")

# === Verificar existencia de archivos requeridos ===
faltantes = []
if not os.path.exists(video_original):
    faltantes.append("Video original")
if not os.path.exists(video_resnet):
    faltantes.append("Video ResNet50")
if not os.path.exists(video_mobilenet):
    faltantes.append("Video MobileNetV3")

if faltantes:
    print(f"❌ Faltan los siguientes archivos para procesar: {', '.join(faltantes)}.")
    exit()

# === Leer CSV y datos ground truth ===
volume_df = pd.read_csv(volume_tracings_csv)
video_df = volume_df[volume_df["FileName"] == video_name]
frames_gt = video_df["Frame"].unique()
print(f"Frames con ground truth: {frames_gt}")

# === Función para área segmentada ===
def get_area(mask):
    return np.sum(mask)

# === Obtener tamaño del frame ===
cap = cv2.VideoCapture(video_original)
ret, frame_sample = cap.read()
cap.release()
frame_height, frame_width = frame_sample.shape[:2]

# === Calcular áreas en cada fuente ===
areas_gt = {}
areas_mobilenet = {}
areas_resnet = {}

for frame_number in frames_gt:
    # --- Ground truth ---
    mask_data = video_df[video_df["Frame"] == frame_number]
    mask_gt = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if not mask_data.empty:
        points = mask_data[["X1", "Y1"]].values.tolist() + \
                 mask_data[["X2", "Y2"]].values.tolist()
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask_gt, [points], 1)
    areas_gt[frame_number] = get_area(mask_gt)

    # --- MobileNet ---
    cap_mob = cv2.VideoCapture(video_mobilenet)
    cap_mob.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_mob = cap_mob.read()
    cap_mob.release()
    mask_mob = (frame_mob[:, :, 2] == 255).astype(np.uint8)
    areas_mobilenet[frame_number] = get_area(mask_mob)

    # --- ResNet ---
    cap_res = cv2.VideoCapture(video_resnet)
    cap_res.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_res = cap_res.read()
    cap_res.release()
    mask_res = (frame_res[:, :, 2] == 255).astype(np.uint8)
    areas_resnet[frame_number] = get_area(mask_res)

# === Calcular EF aproximada ===
frame_ed = min(frames_gt)  # diástole
frame_es = max(frames_gt)  # sístole

def calculate_ef(ed_area, es_area):
    if ed_area == 0:
        return 0.0
    return (ed_area - es_area) / ed_area * 100

ef_gt = calculate_ef(areas_gt[frame_ed], areas_gt[frame_es])
ef_mobilenet = calculate_ef(areas_mobilenet[frame_ed], areas_mobilenet[frame_es])
ef_resnet = calculate_ef(areas_resnet[frame_ed], areas_resnet[frame_es])

# === Mostrar resultados ===
print("\n=== Áreas segmentadas (en pixeles) ===")
print(f"Frame {frame_ed} (EDV) - Ground Truth: {areas_gt[frame_ed]}, MobileNet: {areas_mobilenet[frame_ed]}, ResNet: {areas_resnet[frame_ed]}")
print(f"Frame {frame_es} (ESV) - Ground Truth: {areas_gt[frame_es]}, MobileNet: {areas_mobilenet[frame_es]}, ResNet: {areas_resnet[frame_es]}")

print("\n=== Fracción de eyección aproximada (EF, %) ===")
print(f"Ground Truth EF: {ef_gt:.2f}%")
print(f"MobileNetV3 EF: {ef_mobilenet:.2f}%")
print(f"ResNet50 EF: {ef_resnet:.2f}%")
