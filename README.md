# 🫀 EcoAnalyzer - Segmentación y Análisis de Ecocardiogramasr

**EcoAnalyzer** es una aplicación integral en Python para la segmentación y análisis de videos de ecocardiogramas usando diferentes métodos:  
✅ Ground Truth  
✅ Modelos pre-entrenados (MobileNetV3 y ResNet50)  
✅ U-Net entrenado a partir de las máscaras de segmentación del dataset público **EchoNet-Dynamic**.

🔬 Pensado para uso médico y académico, permite comparar métricas como **IoU, Dice, EDV, ESV y EF** para cada método.

---

## 🗃️ Estructura del Proyecto

ProyectoFinal/

├── data/

│ ├── Videos/ # Videos originales

│ ├── Videos_Output/ # Videos segmentados con ground truth

│ ├── VideosMobileNetV3/ # Videos segmentados con MobileNetV3

│ ├── VideosResNet50/ # Videos segmentados con ResNet50

│ ├── videos_segmentados/ # Videos segmentados con U-Net (y .csv de EF)

├── Dataset_Segmentacion/

│ ├── Datasets.py/ # Código para generar dataset de imágenes y máscaras

│ ├── images/ # (NO subidos por tamaño, se generan con Datasets.py)

│ ├── masks/ # (NO subidos por tamaño, se generan con Datasets.py)

├── Metricas/

│ ├── PruebaMask.py # Ground truth

│ ├── MobileNet.py # MobileNetV3

│ ├── ResNet.py # ResNet50

├── modelo_entrenado/

│ ├── UNet.py # Entrenamiento de U-Net

│ ├── unet_ecocardio.pth # Modelo final entrenado

├── Utils/

│ ├── FileList.csv.zip # Datos del dataset comprimidos

│ ├── VolumeTracings.csv.zip

├── Calculos/

│ ├── CalculoMetricas.py # IoU y Dice

│ ├── CalculosFinales.py # EDV, ESV, EF

├── UNetSegmentacion.py # Segmentación y cálculo de EF con U-Net

├── EcoAnalyzer.py # Aplicación GUI principal

├── informe_final.pdf # Documento final (no incluido)

└── README.md

---

## ⚙️ Dataset

🔗 **Dataset:**  
[EchoNet-Dynamic](https://echonet.github.io/dynamic/)  
Incluye 10,030 videos de ecocardiogramas y segmentaciones expertas.

❗ **Nota:**  
- Solo se incluyeron **~100 videos de prueba**.  
- Las carpetas **`images/`** y **`masks/`** de `Dataset_Segmentacion/` no se subieron.  
- Para recrearlas, ejecuta el script:
```bash
python Dataset_Segmentacion/Datasets.py

🚀 Uso de la Aplicación (EcoAnalyzer)
Lanzar la aplicación principal:

- python EcoAnalyzer.py

Funcionalidades:

📁 Seleccionar video

🟩 Segmentar Ground Truth (PruebaMask.py)

🔵 Segmentar con MobileNetV3

🟠 Segmentar con ResNet50

🔴 Segmentar con U-Net entrenado

🧮 Calcular métricas IoU y Dice

💉 Calcular EDV, ESV y EF (%)

📊 Comparar resultados y exportar CSV

✅ Interfaz amigable y automatizada

Los resultados se guardan automáticamente en las carpetas establecidas según el método.

🧠 Entrenamiento de U-Net
El modelo U-Net fue entrenado con las imágenes y máscaras generadas a partir de las segmentaciones del dataset original.
Para reentrenarlo (opcional):

- python modelo_entrenado/UNet.py

El modelo resultante se guarda en:

-modelo_entrenado/unet_ecocardio.pth

📦 Dependencias
Instalar dependencias principales:

- pip install -r requirements.txt

Incluye:

opencv-python

numpy

torch

torchvision

pandas

scipy

scikit-learn

tkinter (ya viene con Python en Windows)

✏️ Notas adicionales
✔️ Los scripts aceptan el nombre del video como argumento para facilitar la automatización.
✔️ Los resultados parciales de segmentación y métricas ya están incluidos para referencia rápida.

📚 Créditos
📊 Dataset original y publicación:
EchoNet-Dynamic - echonet.github.io/dynamic/

👨‍💻 Desarrollado por:

 - Jhon Edward Ospina Navarro
 - Alvaro Andres Rojas Rojas
 - Jerson Andres Navarrete
 - Mateo Giraldo Zapata
 - Sergio Andres Fernandez

como parte de proyectos académicos en Visión por Computador y Electromedicina.
