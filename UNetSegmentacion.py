import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd

# === Modelo U-Net (igual al entrenado) ===
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.middle = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, 2, stride=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# === Comprobar argumento (nombre del video) ===
if len(sys.argv) < 2:
    print("❌ Error: No se proporcionó el nombre del video.")
    exit()

video_name = sys.argv[1]
video_path = os.path.join("Data", "Videos", video_name)
output_dir = os.path.join("Data", "videos_segmentados")
os.makedirs(output_dir, exist_ok=True)

# === Configuración ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("Modelo_Entrenado/unet_ecocardio.pth", map_location=device))
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor()
])

# === Procesar video ===
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_name_no_ext = video_name.replace(".avi", "")
output_video = os.path.join(output_dir, f"{video_name_no_ext}_unet_segmentado.avi")
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

data = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Procesando video {video_name}...")
for frame_number in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image_resized = cv2.resize(input_image, (256, 256))
    input_tensor = transform(input_image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0, 0].cpu().numpy()
        output_mask = (output > 0.5).astype(np.uint8)

    mask_resized = cv2.resize(output_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    area = np.sum(mask_cleaned)
    data.append({"Frame": frame_number, "Area (px)": area})

    red_mask = np.zeros_like(frame)
    red_mask[:, :, 2] = 255
    frame_segmented = np.where(mask_cleaned[:, :, None] == 1, red_mask, frame)

    out.write(frame_segmented)

    if frame_number % 50 == 0:
        print(f"Procesados {frame_number}/{total_frames} frames...")

cap.release()
out.release()

# === Calcular EDV, ESV y EF ===
df = pd.DataFrame(data)
edv = df["Area (px)"].max()
esv = df["Area (px)"].min()
ef = ((edv - esv) / edv) * 100 if edv != 0 else 0
df["EDV (px)"] = edv
df["ESV (px)"] = esv
df["EF (%)"] = ef

output_csv = os.path.join(output_dir, f"{video_name_no_ext}_areas_ef.csv")
df.to_csv(output_csv, index=False)

# === Resumen final ===
print("\n=== Resumen final ===")
print(f"Frames procesados: {total_frames}")
print(f"EDV: {edv} px²")
print(f"ESV: {esv} px²")
print(f"EF: {ef:.2f} %")
print(f"Video segmentado guardado en: {output_video}")
print(f"Datos exportados a: {output_csv}")
