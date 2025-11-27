# ♻️ proyecto_reciclAI  
Clasificación de residuos (Fase 1: Amarillo, Verde, Azul y Gris) mediante **Deep Learning** usando **EfficientNetB0**, **TensorFlow** y conversión del modelo en **TensorFlow Lite para Android**.

---

## 📌 Descripción  
Este proyecto entrena un modelo utilizando el diseño CNN + transfer learning para lograr una **clasificación de residuos** en categorías:

FASE 1: 
- 🟡 **Amarillo** — Plásticos y envases  
- 🟢 **Verde** — Vidrio  
- 🔵 **Azul** — Papel y cartón
- ⚪ **Gris** — Contenedor genérico (default)

En esta fase solo reconocerá estas tres primeras categorías y si no reconoce el contenedor lo enviará al gris por default.

En próximas fases se agregarán el contenedor marrón (orgánico), productos que van al punto verde (muebles, electrodomesticos), pilas y aceites (contenedores especiales) y material de construccion.

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

```
Dataset_Reciclaje/
 ├── Amarillo/
 ├── Azul/
 └── Verde/
```

---

## 🧠 Modelo: EfficientNetB0
Se utiliza **EfficientNetB0** cargado con pesos de ImageNet en dos fases:

1. ✅ **Transfer Learning** (estructura principal fija, no se entrenan las primeras capas)  
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
Modelo generado TFLite listo para Android:

- `proyecto-reciclAI_quant.tflite` 

**E/S del modelo (formato entrada y salida)**

input : [1, 224, 224, 3] float32 (RGB)
output: 3 probabilidades (softmax)
clases: ["Amarillo", "Verde", "Azul"]

---
```
proyecto_reciclAI/
 ├── proyecto-reciclAI.keras
 ├── proyecto-reciclAI_quant.tflite
 ├── proyecto-reciclAI_fp16.tflite
 ├── loss_curve.png
 ├── accuracy_curve.png
 ├── README.md
 ├── LICENSE
 └── .gitignore
```

🧪 Reproducir el entrenamiento (Colab)
pip install tensorflow matplotlib huggingface_hub


Luego ejecutar el notebook que descarga el dataset desde Hugging Face, entrena (TL + FT) y guarda los modelos.

🧾 Licencia

Este proyecto está bajo MIT License.

✨ Autora

Rita Casiello Pose — IA aplicada a reciclaje y apps móviles.
