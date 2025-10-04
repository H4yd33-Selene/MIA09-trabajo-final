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

