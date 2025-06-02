import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm

# === Dataset personalizado ===
class EchoDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalizar y convertir a tensor
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        image = T.ToTensor()(image)
        mask = torch.from_numpy(mask / 255.).unsqueeze(0).float()

        return image, mask

# === Modelo U-Net básico ===
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

if __name__ == "__main__":
    # === Preparar dataset y dataloader ===
    base_dir = os.path.join(os.getcwd(), "Dataset_Segmentacion")
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")

    dataset = EchoDataset(images_dir, masks_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # === Entrenamiento ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    # === Archivo para mostrar avance en la GUI ===
    with open("progreso_entrenamiento.txt", "w") as f:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for images, masks in tqdm(dataloader, desc=f"Época {epoch+1}/{num_epochs}"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)

            avg_loss = epoch_loss / len(dataset)
            f.write(f"Época {epoch+1}, Pérdida promedio: {avg_loss:.4f}\n")
            f.flush()  # Para que la GUI lo lea en tiempo real
            print(f"Época {epoch+1}, Pérdida promedio: {avg_loss:.4f}")

    # === Guardar modelo entrenado ===
    modelo_dir = os.path.join(os.getcwd(), "modelo_entrenado")
    os.makedirs(modelo_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(modelo_dir, "unet_ecocardio.pth"))
    print("¡Entrenamiento completado y modelo guardado como 'unet_ecocardio.pth'!")
