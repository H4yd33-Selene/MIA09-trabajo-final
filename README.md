# MIA09-trabajo-final
# Visualización Interactiva de Datos (Streamlit)

**Proyecto Final — Maestría en Inteligencia Artificial Aplicada**

Aplicación desarrollada en **Streamlit** que permite la **carga, exploración y visualización interactiva de datasets**.  
El propósito de este proyecto es demostrar la integración de análisis exploratorio de datos (EDA), visualización interactiva y exportación de reportes automatizados en una herramienta web accesible y moderna.

---

## 🎯 Descripción General

Esta aplicación está diseñada para facilitar el análisis y comprensión de datos mediante visualizaciones dinámicas y herramientas estadísticas básicas.  
Entre sus principales funcionalidades se incluyen:

- Carga de archivos en formato **CSV** y **Excel (.xlsx)**.  
- Exploración de datos (vista previa, tipos de columnas, valores nulos y estadísticas descriptivas).  
- Filtros interactivos por columnas numéricas o categóricas.  
- Visualizaciones gráficas con **Plotly**, **Matplotlib** y **Seaborn**:
  - Histogramas  
  - Boxplots  
  - Diagramas de dispersión  
  - Gráficos de barras  
  - Mapas de calor de correlaciones  
- Análisis de Componentes Principales (**PCA**) con visualización interactiva.  
- Exportación de resultados:
  - Dataset filtrado en formato `.csv`
  - Figuras en formato `.png`
  - Reporte completo en formato **PDF** (generado automáticamente con ReportLab)

---

## ⚙️ Requisitos del Sistema

- **Python:** versión 3.8 o superior  
- **Gestor de paquetes:** `pip`

Dependencias principales:  
`streamlit`, `pandas`, `numpy`, `plotly`, `matplotlib`, `seaborn`, `scikit-learn`, `reportlab`

---

## 🧩 Instalación Local

1. Clonar el repositorio desde GitHub:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
2. Crear y activar un entorno virtual (opcional pero recomendado):
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
3. Instalar las dependencias necesarias:
   pip install -r requirements.txt

---

## ▶️ Ejecución de la Aplicación
🔹 Opción 1 — Ejecución local en tu computadora

1. En la raíz del proyecto, ejecuta:
   streamlit run app.py
2. Espera a que se inicie el servidor local y abre el enlace que aparecerá en la terminal:
   Local URL: http://localhost:8501

---

🔹 Opción 2 — Ejecución en Google Colab

Si deseas correr la app directamente en Google Colab, sigue estos pasos:
1. Instala las dependencias:
   !pip install streamlit pyngrok pandas numpy scikit-learn plotly matplotlib seaborn reportlab
2. Configura Ngrok para habilitar el acceso público (requiere token gratuito):
   from pyngrok import ngrok
   !ngrok config add-authtoken 33WpC1TrgkPlR8iOSebYph6YoZ8_3xsdkocSLK1fdABNfVDoT
3. Crea el archivo app.py dentro de Colab:
   %%writefile app.py
   # (pega aquí el código completo de tu aplicación Streamlit)
4. Ejecuta la aplicación:
   public_url = ngrok.connect(8501)
   print("URL pública:", public_url)
   !streamlit run app.py --server.port 8501 &> /dev/null&
5. Abre la URL pública que aparece en la salida — ahí podrás usar la app normalmente.
