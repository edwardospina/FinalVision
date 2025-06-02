import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os

class EcoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("EcoAnalyzer - Análisis y Comparación de Ecocardiogramas")
        master.geometry("600x600")
        master.resizable(False, False)

        self.video_path = ""

        # Rutas automáticas para salidas
        self.data_dir = "data"
        self.output_dirs = {
            "ground_truth": os.path.join(self.data_dir, "Videos_Output"),
            "mobilenet": os.path.join(self.data_dir, "VideosMobileNetV3"),
            "resnet": os.path.join(self.data_dir, "VideosResNet50"),
            "unet": os.path.join(self.data_dir, "videos_segmentados")
        }

        # Título
        tk.Label(master, text="🔬 EcoAnalyzer - Segmentación y Comparación", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Botón y etiqueta para seleccionar video
        tk.Button(master, text="📁 Seleccionar video", command=self.select_video).pack(pady=5)
        self.video_label = tk.Label(master, text="Video no seleccionado", fg="gray")
        self.video_label.pack()

        # Botones para cada módulo
        tk.Label(master, text="🛠️ Opciones de procesamiento:", font=("Helvetica", 12, "bold")).pack(pady=10)

        self.create_buttons_frame()

        # Barra de progreso
        self.progress = ttk.Progressbar(master, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=10)

        # Estado dinámico
        self.status_label = tk.Label(master, text="", font=("Helvetica", 10), fg="blue")
        self.status_label.pack()

        # Cuadro de resultados
        tk.Label(master, text="📊 Resultados:", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.results_text = tk.Text(master, height=10, width=70)
        self.results_text.pack(pady=10)

        # Botón de salir
        tk.Button(master, text="❌ Salir", command=master.quit, bg="red", fg="white").pack(pady=5)

        # Pie de página
        tk.Label(master, text="Versión 1.0 - EcoAnalyzer", font=("Helvetica", 8), fg="gray").pack(side="bottom", pady=5)

    def create_buttons_frame(self):
        frame = tk.Frame(self.master)
        frame.pack()

        tk.Button(frame, text="🔧 Generar Dataset", command=self.generar_dataset).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(frame, text="🎥 Ground Truth Video", command=self.procesar_ground_truth).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(frame, text="🎥 MobileNet Video", command=self.procesar_mobilenet).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(frame, text="🎥 ResNet Video", command=self.procesar_resnet).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(frame, text="🧠 Entrenar U-Net", command=self.entrenar_unet).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(frame, text="🎥 Segmentar con U-Net", command=self.segmentar_unet).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(frame, text="📈 Calcular Métricas (IoU, Dice)", command=self.calcular_metricas).grid(row=3, column=0, padx=5, pady=5)
        tk.Button(frame, text="📈 Calcular EF, EDV, ESV", command=self.calcular_ef).grid(row=3, column=1, padx=5, pady=5)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.avi *.mp4")])
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"📹 Video: {file_path}", fg="black")

    # === Funciones para los botones (vacías por ahora, las llenamos después) ===

    def generar_dataset(self):
        self.status_label.config(text="⚙️ Generando dataset (puede tardar un poco)...")
        self.master.update_idletasks()

        try:
            # Llama directamente a tu script Datasets.py
            resultado = subprocess.run(["python", "Dataset_Segmentacion/Datasets.py"], capture_output=True, text=True)

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Dataset generado exitosamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ Dataset completo.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error al generar dataset:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en la generación de dataset.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def procesar_ground_truth(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video antes de procesar.")
            return

        self.status_label.config(text="⚙️ Procesando Ground Truth (puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess
        import os

        # Nombre del archivo (solo nombre, no ruta completa)
        video_name = os.path.basename(self.video_path)

        try:
            resultado = subprocess.run(
                ["python", "Metricas/PruebaMask.py", video_name],
                capture_output=True,
                text=True
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Video con Ground Truth generado correctamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ Ground Truth completado.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error en Ground Truth:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en Ground Truth.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def procesar_mobilenet(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video antes de procesar.")
            return

        self.status_label.config(text="⚙️ Procesando con MobileNetV3 (puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess
        import os

        # Nombre del archivo (solo nombre, no ruta completa)
        video_name = os.path.basename(self.video_path)

        try:
            resultado = subprocess.run(
                ["python", "Metricas/MobileNet.py", video_name],
                capture_output=True,
                text=True
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Video procesado con MobileNetV3 correctamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ MobileNetV3 completado.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error en MobileNetV3:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en MobileNetV3.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def procesar_resnet(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video antes de procesar.")
            return

        self.status_label.config(text="⚙️ Procesando con ResNet50 (puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess
        import os

        # Nombre del archivo (solo nombre, no ruta completa)
        video_name = os.path.basename(self.video_path)

        try:
            resultado = subprocess.run(
                ["python", "Metricas/ResNet.py", video_name],
                capture_output=True,
                text=True
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Video procesado con ResNet50 correctamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ ResNet50 completado.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error en ResNet50:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en ResNet50.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def entrenar_unet(self):
        self.status_label.config(text="⚙️ Entrenando U-Net (esto puede tardar bastante tiempo)...")
        self.master.update_idletasks()

        import subprocess
        import threading
        import time
        import os

        # Borrar el archivo de progreso si ya existe
        if os.path.exists("progreso_entrenamiento.txt"):
            os.remove("progreso_entrenamiento.txt")

        # Lanzar el entrenamiento en un hilo separado
        def entrenamiento():
            try:
                proceso = subprocess.Popen(
                    ["python", "modelo_entrenado/UNet.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Mientras entrena, lee el archivo de progreso y actualiza la GUI
                while proceso.poll() is None:
                    if os.path.exists("progreso_entrenamiento.txt"):
                        with open("progreso_entrenamiento.txt", "r") as f:
                            progreso = f.read()
                            self.results_text.delete(1.0, tk.END)
                            self.results_text.insert(tk.END, progreso)
                    time.sleep(2)

                # Al terminar, muestra salida completa
                stdout, stderr = proceso.communicate()
                if proceso.returncode == 0:
                    self.status_label.config(text="✅ Entrenamiento completado.")
                    self.results_text.insert(tk.END, "\n✅ Entrenamiento finalizado correctamente.\n")
                    self.results_text.insert(tk.END, stdout)
                else:
                    self.status_label.config(text="❌ Error en entrenamiento.")
                    self.results_text.insert(tk.END, "\n❌ Error durante el entrenamiento:\n")
                    self.results_text.insert(tk.END, stderr)

            except Exception as e:
                self.status_label.config(text="❌ Error inesperado.")
                self.results_text.insert(tk.END, f"\n❌ Error inesperado: {str(e)}")

        # Iniciar el hilo
        threading.Thread(target=entrenamiento).start()

    def segmentar_unet(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video para segmentar con U-Net.")
            return

        video_name = os.path.basename(self.video_path)

        self.status_label.config(text="⚙️ Segmentando con U-Net (esto puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess

        try:
            resultado = subprocess.run(
                ["python", "UNetSegmentacion.py", video_name],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Segmentación con U-Net completada.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ Segmentación finalizada.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error durante la segmentación:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en la segmentación.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def calcular_metricas(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video antes de calcular métricas.")
            return

        video_name = os.path.basename(self.video_path)
        video_name_no_ext = video_name.replace('.avi', '')

        # Verificar existencia de los videos de salida
        resnet_output = os.path.join("data", "VideosResNet50", f"{video_name_no_ext}_resnet.avi")
        mobilenet_output = os.path.join("data", "VideosMobileNetV3", f"{video_name_no_ext}_mobilenet.avi")

        falta = []
        if not os.path.exists(resnet_output):
            falta.append("ResNet50")
        if not os.path.exists(mobilenet_output):
            falta.append("MobileNetV3")

        if falta:
            msg = f"No se encontró salida de: {', '.join(falta)}.\n\nPor favor, genera primero estos resultados."
            messagebox.showwarning("Archivos faltantes", msg)
            return

        self.status_label.config(text="⚙️ Calculando métricas (puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess

        try:
            resultado = subprocess.run(
                ["python", "Calculos/CalculoMetricas.py", video_name],
                capture_output=True,
                text=True
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ Métricas calculadas correctamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ Métricas finalizadas.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error en cálculo de métricas:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en cálculo de métricas.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

    def calcular_ef(self):
        if not self.video_path:
            messagebox.showwarning("Atención", "Por favor selecciona un video antes de calcular EF.")
            return

        video_name = os.path.basename(self.video_path)
        video_name_no_ext = video_name.replace('.avi', '')

        # Verificar existencia de los videos de salida
        original_video = os.path.join("data", "Videos", video_name)
        resnet_output = os.path.join("data", "VideosResNet50", f"{video_name_no_ext}_resnet.avi")
        mobilenet_output = os.path.join("data", "VideosMobileNetV3", f"{video_name_no_ext}_mobilenet.avi")

        faltantes = []
        if not os.path.exists(original_video):
            faltantes.append("Video original")
        if not os.path.exists(resnet_output):
            faltantes.append("ResNet50")
        if not os.path.exists(mobilenet_output):
            faltantes.append("MobileNetV3")

        if faltantes:
            msg = f"No se encontró: {', '.join(faltantes)}.\n\nPor favor, genera estos resultados primero."
            messagebox.showwarning("Archivos faltantes", msg)
            return

        self.status_label.config(text="⚙️ Calculando EF (esto puede tardar un poco)...")
        self.master.update_idletasks()

        import subprocess

        try:
            resultado = subprocess.run(
                ["python", "Calculos/CalculosFinales.py", video_name],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

            if resultado.returncode == 0:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✅ EF y áreas calculadas correctamente.\n")
                self.results_text.insert(tk.END, resultado.stdout)
                self.status_label.config(text="✅ Cálculo de EF finalizado.")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "❌ Error en cálculo de EF:\n")
                self.results_text.insert(tk.END, resultado.stderr)
                self.status_label.config(text="❌ Error en cálculo de EF.")

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error inesperado: {str(e)}")
            self.status_label.config(text="❌ Error inesperado.")

# === Lanzar app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = EcoAnalyzerApp(root)
    root.mainloop()
