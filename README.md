# ♻️ proyecto_reciclAI  
Clasificación de residuos (Amarillo, Verde y Azul) mediante **Deep Learning** usando **EfficientNetB0**, **TensorFlow** y despliegue del modelo en **TensorFlow Lite para Android**.

---

## 📌 Descripción  
Este proyecto entrena un modelo de visión por computadora capaz de **clasificar residuos** en tres categorías:

- 🟡 **Amarillo** — Plásticos y envases  
- 🟢 **Verde** — Vidrio  
- 🔵 **Azul** — Papel y cartón  

El objetivo final es integrar este modelo en una **aplicación Android** que ayude a los usuarios a reciclar correctamente utilizando solo la cámara del móvil.

---

## 🚀 Tecnologías utilizadas
- **TensorFlow / Keras**
- **EfficientNetB0** (Transfer Learning + Fine-Tuning)
- **TensorFlow Lite** para despliegue en móviles
- **Google Colab**
- **GitHub** para versionado
- **Matplotlib** para gráficas
- **Hugging Face Hub** para almacenar el dataset original

---

## 📊 Dataset  
El dataset utilizado proviene de mi cuenta de Hugging Face:

👉 https://huggingface.co/datasets/Ritz-ai/dataset-reciclaje  

Estructura del dataset:

Dataset_Reciclaje/
├── Amarillo/
├── Azul/
└── Verde/


---

## 🧠 Modelo: EfficientNetB0
Se utiliza **EfficientNetB0** cargado con pesos de ImageNet en dos fases:

1. ✅ **Transfer Learning** (backbone congelado)  
2. ✅ **Fine-Tuning** (últimas capas descongeladas)

El modelo final incluye:
- Preprocesado interno (`preprocess_input`)
- GlobalAveragePooling2D
- Dropout (0.2)
- Dense softmax (3 clases)

---

## ✨ Data Augmentation  
- `RandomFlip("horizontal")`  
- `RandomRotation(0.05)`  
- `RandomZoom(0.1)`  
- `RandomContrast(0.1)`

---

## 📈 Resultados
- **Accuracy validación ≈ 0.92**
- Pérdida de validación descendente y estable
- Sin sobreajuste apreciable

Gráficas generadas:

loss_curve.png
accuracy_curve.png

---

## 📱 Exportación a TensorFlow Lite
Se generan dos modelos TFLite listos para Android:

- `proyecto-reciclAI_quant.tflite` → Dynamic Range Quantization  
- `proyecto-reciclAI_fp16.tflite` → Float16 (ideal para GPU delegate)

**E/S del modelo**

input : [1, 224, 224, 3] float32 (RGB)
output: 3 probabilidades (softmax)
clases: ["Amarillo", "Verde", "Azul"]

---

## 📱 Uso en Android (resumen)
```kotlin
// Ejemplo con TensorFlow Lite + TensorImage (Support Library)
val tflite = Interpreter(loadModelFile("proyecto-reciclAI_fp16.tflite"))
val image = TensorImage(DataType.FLOAT32).apply { load(bitmap) }
val processor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .build()
val input = processor.process(image).buffer
val output = Array(1) { FloatArray(3) }
tflite.run(input, output)
val clase = arrayOf("Amarillo","Verde","Azul")[output[0].indices.maxBy { output[0][it] }!!]

proyecto_reciclAI/
 ├── proyecto-reciclAI.keras
 ├── proyecto-reciclAI_quant.tflite
 ├── proyecto-reciclAI_fp16.tflite
 ├── loss_curve.png
 ├── accuracy_curve.png
 ├── README.md
 ├── LICENSE
 └── .gitignore

🧪 Reproducir el entrenamiento (Colab)
pip install tensorflow matplotlib huggingface_hub


Luego ejecutar el notebook que descarga el dataset desde Hugging Face, entrena (TL + FT) y guarda los artefactos.

🧾 Licencia

Este proyecto está bajo MIT License.

✨ Autora

Rituzka — IA aplicada a reciclaje y apps móviles.
