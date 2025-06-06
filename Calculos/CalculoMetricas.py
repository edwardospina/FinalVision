import cv2
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import jaccard_score, f1_score

# Obtener nombre del video
if len(sys.argv) > 1:
    video_name = sys.argv[1]
else:
    print("❌ Error: No se proporcionó el nombre del video.")
    exit()

# Rutas relativas
video_resnet_path = os.path.join("data", "VideosResNet50", f"{video_name.replace('.avi', '')}_resnet.avi")
video_mobilenet_path = os.path.join("data", "VideosMobileNetV3", f"{video_name.replace('.avi', '')}_mobilenet.avi")
volume_tracings_csv = os.path.join("utils", "VolumeTracings.csv")

# Leer CSV
volume_df = pd.read_csv(volume_tracings_csv)
video_df = volume_df[volume_df["FileName"] == video_name]

# Frames únicos con ground truth
unique_frames = video_df["Frame"].unique()
print(f"Frames con ground truth: {unique_frames}")

# === Funciones ===
def get_ground_truth_mask(frame_number, frame_width, frame_height):
    mask_data = video_df[video_df["Frame"] == frame_number]
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if not mask_data.empty:
        points = mask_data[["X1", "Y1"]].values.tolist() + \
                 mask_data[["X2", "Y2"]].values.tolist()
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def get_predicted_mask(frame, color_channel=0):
    mask = frame[:, :, color_channel]
    mask_binary = (mask > 127).astype(np.uint8)
    return mask_binary

def compute_metrics(gt_mask, pred_mask):
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    return iou, dice

def evaluate_video(video_path, frame_width, frame_height):
    cap = cv2.VideoCapture(video_path)
    iou_scores = []
    dice_scores = []
    for frame_number in unique_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        gt_mask = get_ground_truth_mask(frame_number, frame_width, frame_height)
        pred_mask = get_predicted_mask(frame, color_channel=0)

        iou, dice = compute_metrics(gt_mask, pred_mask)
        iou_scores.append(iou)
        dice_scores.append(dice)

        print(f"Frame {frame_number}: IoU={iou:.4f}, Dice={dice:.4f}")

    cap.release()
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    return mean_iou, mean_dice

# Tamaño de frame
cap_sample = cv2.VideoCapture(video_resnet_path)
ret, sample_frame = cap_sample.read()
cap_sample.release()
frame_height, frame_width = sample_frame.shape[:2]

# Evaluar
print("\n=== Evaluando video - ResNet50 ===")
mean_iou_resnet, mean_dice_resnet = evaluate_video(video_resnet_path, frame_width, frame_height)

print("\n=== Evaluando video - MobileNetV3 ===")
mean_iou_mobilenet, mean_dice_mobilenet = evaluate_video(video_mobilenet_path, frame_width, frame_height)

# Resultados finales
print("\n=== Resultados finales (solo frames con ground truth) ===")
print(f"ResNet50 - Mean IoU: {mean_iou_resnet:.4f}, Mean Dice: {mean_dice_resnet:.4f}")
print(f"MobileNetV3 - Mean IoU: {mean_iou_mobilenet:.4f}, Mean Dice: {mean_dice_mobilenet:.4f}")
