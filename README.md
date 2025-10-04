# MIA09-trabajo-final
# Visualización Interactiva de Datos (Streamlit)

**Proyecto final** — Aplicación en Streamlit para carga, exploración y visualización interactiva de datasets.

## Descripción
Aplicación en **Streamlit** que permite a usuarios:
- Cargar datasets (CSV / Excel).
- Realizar análisis exploratorio (EDA): vista previa, tipos, nulos y estadísticas.
- Generar visualizaciones interactivas: histogramas, boxplots, scatter, barras, heatmap de correlación.
- Ejecutar PCA interactivo.
- Exportar resultados: dataset filtrado (.csv), figuras (.png) y reporte PDF con resumen y gráficas.

## Funcionalidades principales
- Carga de `.csv` y `.xlsx`.
- Estadísticas descriptivas y manejo básico de nulos.
- Filtros dinámicos por columnas (numéricas y categóricas).
- Visualizaciones con Plotly y Matplotlib/Seaborn.
- Generación automática de reporte PDF (reportlab).
- Descarga de dataset procesado y de figuras.

## Requisitos
- Python 3.8+
- pip

## Instalación local
```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
