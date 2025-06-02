import cv2
import pandas as pd
import numpy as np
import os
import sys

# === Rutas relativas ===
base_dir = os.getcwd()
videos_folder = os.path.join(base_dir, "data", "Videos")
filelist_csv = os.path.join(base_dir, "utils", "FileList.csv")
volume_tracings_csv = os.path.join(base_dir, "utils", "VolumeTracings.csv")
output_folder = os.path.join(base_dir, "data", "Videos_Output")
os.makedirs(output_folder, exist_ok=True)

# === Obtener nombre del video de los argumentos ===
if len(sys.argv) > 1:
    sample_video_name = sys.argv[1]
else:
    print("❌ Error: No se proporcionó el nombre del video.")
    exit()

video_path = os.path.join(videos_folder, sample_video_name)
print(f"Usando video: {video_path}")

# === Abrir video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === VideoWriter ===
output_video_path = os.path.join(output_folder, f"{sample_video_name.replace('.avi', '')}_annotated.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# === Procesar frames ===
frame_number = 0
volume_tracings_df = pd.read_csv(volume_tracings_csv)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask_data = volume_tracings_df[
        (volume_tracings_df["FileName"] == sample_video_name) &
        (volume_tracings_df["Frame"] == frame_number)
    ]

    if not mask_data.empty:
        points = mask_data[["X1", "Y1"]].values.tolist() + \
                 mask_data[["X2", "Y2"]].values.tolist()
        points = np.array(points, dtype=np.int32)

        mask_binary = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask_binary, [points], 1)

        red_mask = np.zeros_like(frame_rgb)
        red_mask[:, :, 2] = 255

        annotated_frame = np.where(mask_binary[:, :, None] == 1, red_mask, frame_rgb)
    else:
        annotated_frame = frame_rgb

    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    out.write(annotated_frame_bgr)

    frame_number += 1
    if frame_number % 50 == 0:
        print(f"Procesados {frame_number}/{num_frames} frames...")

print("✅ Video anotado generado correctamente.")
print(f"Guardado en: {output_video_path}")

cap.release()
out.release()
