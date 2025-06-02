import cv2
import pandas as pd
import numpy as np
import os

# === Rutas (relativas a la carpeta donde se ejecuta este script) ===
base_dir = os.getcwd()  # Carpeta actual
videos_folder = os.path.join(base_dir, "Data", "Videos")
volume_tracings_csv = os.path.join(base_dir, "utils", "VolumeTracings.csv")

# Carpeta donde guardaremos dataset
dataset_dir = os.path.join(base_dir, "Dataset_Segmentacion")
images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Leer CSV
volume_df = pd.read_csv(volume_tracings_csv)
video_names = volume_df["FileName"].unique()

frame_count = 0

# Recorremos todos los videos con ground truth
for video_name in video_names:
    video_path = os.path.join(videos_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir: {video_name}")
        continue

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames_gt = volume_df[volume_df["FileName"] == video_name]["Frame"].unique()

    for frame_number in frames_gt:
        # Leer el frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        # Guardar la imagen original
        image_filename = f"{video_name}_{frame_number:04d}.png"
        cv2.imwrite(os.path.join(images_dir, image_filename), frame)

        # Generar la máscara binaria
        mask_data = volume_df[
            (volume_df["FileName"] == video_name) &
            (volume_df["Frame"] == frame_number)
        ]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        if not mask_data.empty:
            points = mask_data[["X1", "Y1"]].values.tolist() + \
                     mask_data[["X2", "Y2"]].values.tolist()
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)  # máscara blanca en el área segmentada

        # Guardar la máscara
        mask_filename = f"{video_name}_{frame_number:04d}.png"
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Guardados {frame_count} pares imagen-máscara...")

    cap.release()

print("¡Dataset de segmentación generado exitosamente!")
print(f"Total de pares imagen-máscara: {frame_count}")
print(f"Imágenes: {images_dir}")
print(f"Máscaras: {masks_dir}")
