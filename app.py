import io
import base64
from typing import Tuple
import os
import seaborn as sns
import tempfile
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

st.set_page_config(page_title="Visualización de Datos (Streamlit)", layout="wide")

# ----------------------
# Helper utilities
# ----------------------
@st.cache_data
def load_example(name: str) -> pd.DataFrame:
    if name == "Iris (sklearn)":
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        # rename target column for clarity
        df.rename(columns={'target': 'species'}, inplace=True)
        return df
    elif name == "Ejemplo pequeño (tips - seaborn)":
        import seaborn as sns
        return sns.load_dataset("tips")
    else:
        return pd.DataFrame()

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato de archivo no soportado. Use CSV o Excel.")

def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def download_df_as_csv(df: pd.DataFrame, filename: str = "data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="🔽 Descargar CSV", data=csv, file_name=filename, mime='text/csv')

def fig_to_bytes(plt_fig):
    buf = io.BytesIO()
    plt_fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf
def generate_pdf_report(df_summary_text, df_head, stats_df, img_bufs, output_filename="reporte_analisis.pdf"):
    """Genera un PDF con resumen, estadísticas y gráficos."""
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, output_filename)
    c = canvas.Canvas(out_path, pagesize=landscape(letter))
    width, height = landscape(letter)

    # Título
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 40, "Reporte de Análisis de Datos (Streamlit)")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 60, "Generado automáticamente")

    # Resumen
    text_y = height - 90
    c.setFont("Helvetica", 9)
    for line in df_summary_text.split("\n"):
        c.drawString(40, text_y, line)
        text_y -= 12
        if text_y < 100:
            c.showPage()
            text_y = height - 60

    # Estadísticas
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 40, "Estadísticas descriptivas")
    y = height - 70
    c.setFont("Helvetica", 8)
    for col in stats_df.columns[:8]:
        c.drawString(40 + stats_df.columns.get_loc(col)*90, y, str(col))
    y -= 14
    for idx, row in stats_df.iterrows():
        c.drawString(10, y, str(idx))
        for j, col in enumerate(stats_df.columns[:8]):
            c.drawString(40 + j*90, y, str(round(row[col], 3)))
        y -= 12
        if y < 100:
            c.showPage()
            y = height - 40

    # Agregar imágenes
    for name, buf in img_bufs.items():
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, height - 40, name)
        try:
            img = ImageReader(buf)
            iw, ih = img.getSize()
            scale = min((width-80)/iw, (height-100)/ih)
            c.drawImage(img, 40, 60, iw*scale, ih*scale)
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(40, height - 80, f"Error al insertar imagen: {e}")

    c.save()
    return out_path

# ----------------------
# Sidebar: carga y opciones globales
# ----------------------
st.sidebar.title("Controles")
st.sidebar.markdown("Sube un dataset (.csv / .xlsx) o selecciona un dataset de ejemplo.")

file = st.sidebar.file_uploader("Subir archivo (CSV / XLSX)", type=["csv", "xlsx"])
use_example = st.sidebar.selectbox("Dataset de ejemplo", ["-- ninguno --", "Iris (sklearn)", "Ejemplo pequeño (tips - seaborn)"])
if use_example != "-- ninguno --":
    df = load_example(use_example)
else:
    df = read_uploaded_file(file)

st.sidebar.markdown("---")
st.sidebar.markdown("Opciones de preprocesamiento:")
if not df.empty:
    st.sidebar.write(f"Filas: {df.shape[0]} — Columnas: {df.shape[1]}")
    na_strategy = st.sidebar.selectbox("Imputación de nulos", ["Sin acción", "Eliminar filas con nulos", "Imputar mediana (num) / modo (cat)"])
else:
    na_strategy = "Sin acción"

st.title("📊 Visualización Interactiva de Datos")
st.write("Una aplicación para cargar datasets, explorar y visualizar datos. Está pensada para tareas, investigación y presentaciones.")

# ----------------------
# Main: cuando no hay datos
# ----------------------
if df.empty:
    st.info("Carga un archivo CSV/XLSX o selecciona un dataset de ejemplo desde la barra lateral para comenzar.")
    st.caption("Consejo: usa el dataset 'Iris (sklearn)' para pruebas rápidas.")
    st.stop()

# ----------------------
# Preprocesamiento simple
# ----------------------
df_original = df.copy()

if na_strategy == "Eliminar filas con nulos":
    df = df.dropna()
elif na_strategy == "Imputar mediana (num) / modo (cat)":
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    for c in num_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)
    for c in cat_cols:
        mode = df[c].mode()
        if not mode.empty:
            df[c] = df[c].fillna(mode[0])

# ----------------------
# EDA: vista y stats
# ----------------------
st.header("1. Exploración de Datos")
with st.expander("Vista rápida del dataset"):
    st.subheader("Datos (primeras filas)")
    st.dataframe(df.head())
    st.markdown(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
with st.expander("Tipos y valores nulos"):
    st.write("Tipos de columnas:")
    st.write(df.dtypes)
    st.write("Valores nulos por columna:")
    st.write(df.isnull().sum())

with st.expander("Estadísticas descriptivas (numéricas)"):
    st.write(df.describe().T)

# ----------------------
# Interactividad: filtros
# ----------------------
st.header("2. Filtros y segmentación")
cols_for_filters = st.multiselect("Selecciona columnas para filtrar (categorical / numeric)", df.columns.tolist(), default=[])
df_filtered = df.copy()
if cols_for_filters:
    st.write("Aplique filtros:")
    for col in cols_for_filters:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            vals = df[col].unique().tolist()
            sel = st.multiselect(f"{col} (categorical)", options=vals, default=vals)
            df_filtered = df_filtered[df_filtered[col].isin(sel)]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            r = st.slider(f"{col} (numeric range)", min_val, max_val, (min_val, max_val))
            df_filtered = df_filtered[(df_filtered[col] >= r[0]) & (df_filtered[col] <= r[1])]

st.write(f"Filtrado: {df_filtered.shape[0]} filas seleccionadas")
if st.button("🔁 Reset filtros"):
    df_filtered = df.copy()

# ----------------------
# Visualizaciones interactivas (Plotly)
# ----------------------
st.header("3. Visualizaciones Interactivas")

numeric_columns = get_numeric_columns(df_filtered)
categorical_columns = get_categorical_columns(df_filtered)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma")
    hist_col = st.selectbox("Variable", numeric_columns, index=0 if numeric_columns else None, key="hist")
    bins = st.slider("Bins", 5, 100, 20)
    if hist_col:
        fig = px.histogram(df_filtered, x=hist_col, nbins=bins, marginal="box", title=f"Histograma de {hist_col}")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Boxplot")
    box_col = st.selectbox("Variable (boxplot)", numeric_columns, index=0 if numeric_columns else None, key="box")
    if box_col:
        fig = px.box(df_filtered, y=box_col, points="all", title=f"Boxplot de {box_col}")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Diagrama de dispersión (Scatter)")
x_axis = st.selectbox("Eje X", numeric_columns, index=0, key="scatter_x")
y_axis = st.selectbox("Eje Y", numeric_columns, index=1 if len(numeric_columns) > 1 else 0, key="scatter_y")
color_by = st.selectbox("Color por (opc.)", [None] + df_filtered.columns.tolist(), index=0, key="scatter_color")
size_by = st.selectbox("Tamaño por (opc.)", [None] + numeric_columns, index=0, key="scatter_size")

if x_axis and y_axis:
    fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_by if color_by else None,
                     size=size_by if size_by else None, trendline="ols", title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Gráfico de barras (conteo por categoría)")
if categorical_columns:
    cat_col = st.selectbox("Columna categórica", categorical_columns, key="bar_cat")
    if cat_col:
        counts = df_filtered[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.bar(counts, x=cat_col, y="count", title=f"Conteo por {cat_col}")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay columnas categóricas detectadas para gráfico de barras.")

# ----------------------
# Correlación - Heatmap
# ----------------------
st.header("4. Matriz de correlación")
if len(numeric_columns) >= 2:
    corr = df_filtered[numeric_columns].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Mapa de calor: correlaciones (numéricas)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Se requieren al menos 2 columnas numéricas para matriz de correlación.")

# ----------------------
# PCA interactivo
# ----------------------
st.header("5. Análisis de Componentes Principales (PCA)")
if len(numeric_columns) >= 2:
    n_comp = st.slider("Número de componentes PCA (visualización)", 2, min( min(len(numeric_columns), 10), len(numeric_columns) ), value=2)
    do_pca = st.checkbox("Ejecutar PCA", value=True)
    if do_pca:
        X = df_filtered[numeric_columns].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=n_comp)
        Xp = pca.fit_transform(Xs)
        explained = pca.explained_variance_ratio_
        cum_explained = np.cumsum(explained)

        st.write("Varianza explicada por componente:", np.round(explained, 4))
        st.write("Varianza acumulada:", np.round(cum_explained, 4))

        if n_comp >= 2:
            pc_x = st.selectbox("Componente eje X", range(1, n_comp+1), index=0, key="pcx")
            pc_y = st.selectbox("Componente eje Y", range(1, n_comp+1), index=1 if n_comp >= 2 else 0, key="pcy")
            color_pca = st.selectbox("Colorear por (opcional)", [None] + df_filtered.columns.tolist(), index=0, key="pcacolor")

            df_pca = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(n_comp)], index=X.index)
            if color_pca:
                color_vals = df_filtered.loc[df_pca.index, color_pca]
            else:
                color_vals = None

            fig = px.scatter(df_pca, x=f"PC{pc_x}", y=f"PC{pc_y}", color=color_vals, title=f"PCA - PC{pc_x} vs PC{pc_y}")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 columnas numéricas para aplicar PCA.")

# ----------------------
# Exportar datos / figuras
# ----------------------
st.header("6. Exportar datos y gráficos")
with st.expander("Descargar dataset filtrado / procesado"):
    download_df_as_csv(df_filtered, filename="dataset_filtrado.csv")

with st.expander("Exportar figura actual (matplotlib)"):
    st.write("Si deseas descargar una figura específica en PNG, presiona el botón para generar y descargar.")
    if st.button("Generar figura PNG de ejemplo (histograma)"):
        if hist_col:
            fig_mat, ax = plt.subplots()
            ax.hist(df_filtered[hist_col].dropna(), bins=bins)
            ax.set_title(f"Histograma (matplotlib) - {hist_col}")
            buf = fig_to_bytes(fig_mat)
            st.download_button("Descargar PNG", data=buf, file_name=f"hist_{hist_col}.png", mime="image/png")
            plt.close(fig_mat)
        else:
            st.info("No hay variable seleccionada para histograma.")
# ----------------------
# Generar reporte PDF
# ----------------------
st.subheader("📄 Generar Reporte PDF")

# Crear buffers con imágenes
img_bufs = {}

# Histograma
if 'hist_col' in locals() and hist_col:
    fig, ax = plt.subplots()
    ax.hist(df_filtered[hist_col].dropna(), bins=bins)
    ax.set_title(f"Histograma - {hist_col}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_bufs[f"Histograma_{hist_col}"] = buf
    plt.close(fig)

# Heatmap
if len(numeric_columns) >= 2:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df_filtered[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", bbox_inches='tight')
    buf2.seek(0)
    img_bufs["Heatmap_correlaciones"] = buf2
    plt.close(fig)

# PCA
try:
    if 'df_pca' in locals():
        fig = px.scatter(df_pca, x='PC1', y='PC2')
        img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
        buf3 = io.BytesIO(img_bytes)
        img_bufs["PCA_proyeccion"] = buf3
except Exception:
    pass

# Resumen textual
summary_text = f"Dataset: {df_filtered.shape[0]} filas x {df_filtered.shape[1]} columnas\n" \
               f"Nulos por columna:\n{df_filtered.isnull().sum().to_string()}"

# Botón para generar PDF
if st.button("📄 Generar reporte PDF"):
    stats_df = df_filtered.describe()
    df_head = df_filtered.head(10)
    pdf_path = generate_pdf_report(summary_text, df_head, stats_df, img_bufs)
    with open(pdf_path, "rb") as f:
        st.download_button("🔽 Descargar Reporte PDF", data=f, file_name="reporte_analisis.pdf", mime="application/pdf")

# ----------------------
# Sección de ayuda / documentación
# ----------------------
st.markdown("---")
st.subheader("Ayuda y notas")
st.markdown("""
- Formatos soportados: **CSV** y **Excel (.xls/.xlsx)**.
- Para mejores resultados en PCA y correlaciones, trabaje con columnas numéricas y realice imputación si hay nulos.
- Use la opción de filtros para generar subconjuntos de interés antes de graficar.
- Para desplegar la app públicamente puede usar [Streamlit Community Cloud](https://streamlit.io/cloud) o desplegar en un servidor con `streamlit run app.py`.
""")
